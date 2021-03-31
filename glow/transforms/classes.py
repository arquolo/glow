__all__ = [
    'ChannelMix', 'ChannelShuffle', 'CutOut', 'BitFlipNoise', 'Elastic',
    'LumaJitter', 'DegradeJpeg', 'DegradeQuality', 'FlipAxis', 'HsvShift',
    'MaskDropout', 'MultiNoise', 'WarpAffine'
]

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.stats import ortho_group

from .core import Transform

# ---------------------------------- mixins ----------------------------------


class _LutMixin(Transform):
    def apply(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        assert image.dtype == np.uint8
        if image.ndim == 2:
            return cv2.LUT(image, lut)

        luts = lut if lut.ndim == 2 else [lut] * image.shape[-1]
        planes = [
            cv2.LUT(plane, lut_)
            for plane, lut_ in zip(cv2.split(image), luts)
        ]
        return cv2.merge(planes).reshape(image.shape)


class _InlineRandom(Transform):
    def prepare(self, rg: np.random.Generator, **data) -> dict[str, Any]:
        return {'rg': rg}


# ---------------------------------- noise ----------------------------------


@dataclass
class MultiNoise(Transform):
    low: float = 0.5
    high: float = 1.5

    def prepare(self, rg: np.random.Generator, image: np.ndarray, **_):
        assert image.dtype == np.uint8

        scale = rg.random(image.shape[:2], dtype='f4')
        scale *= (self.high - self.low)
        scale -= self.low
        return {'scale': scale[..., None]}

    def apply(self, image: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (image * scale).clip(0, 255).astype('u1')


@dataclass
class BitFlipNoise(_InlineRandom):
    kept_planes: int = 4

    def apply(self, image: np.ndarray, rg: np.random.Generator) -> np.ndarray:
        assert image.dtype.kind == 'u'
        planes = 8 * image.dtype.itemsize

        if self.kept_planes >= planes:
            return image

        high_flip = 1 << (planes - self.kept_planes)
        bitmask = (1 << planes) - high_flip
        return ((image & bitmask) +
                rg.integers(high_flip, size=image.shape, dtype=image.dtype))


# ----------------------------- color alteration -----------------------------


class ChannelShuffle(_InlineRandom):
    def apply(self, image: np.ndarray, rg: np.random.Generator) -> np.ndarray:
        assert image.ndim == 3
        return image[:, :, rg.permutation(image.shape[-1])]


class ChannelMix(_InlineRandom):
    intensity: tuple[float, float] = (0.5, 1.5)

    def apply(self, image: np.ndarray, rg: np.random.Generator) -> np.ndarray:
        assert image.ndim == 3
        assert image.dtype == np.uint8
        image = image.astype('f4')

        num_channels = image.shape[-1]
        mat = ortho_group.rvs(num_channels, random_state=rg).astype('f4')

        mat *= rg.uniform(*self.intensity)
        lumat = np.full((num_channels, num_channels), 1 / num_channels)

        image = image @ ((np.eye(num_channels) - lumat) @ mat + lumat)

        return image.clip(0, 255).astype('u1')  # type: ignore


@dataclass
class LumaJitter(_LutMixin):
    brightness: tuple[float, float] = (-0.2, 0.2)
    contrast: tuple[float, float] = (0.8, 1.2)

    def prepare(self, rg: np.random.Generator, **_) -> dict[str, Any]:
        lut = np.arange(256, dtype='f4')

        lut += 256 * rg.uniform(*self.brightness)
        lut = (lut - 128) * rg.uniform(*self.contrast) + 128

        return {'lut': lut.clip(0, 255).astype('u1')}


@dataclass
class HsvShift(_LutMixin):
    max_shift: int = 20

    def prepare(self, rg: np.random.Generator, image: np.ndarray,
                **_) -> dict[str, Any]:
        hue, sat, val = rg.uniform(-self.max_shift, self.max_shift, size=3)
        lut = np.arange(256, dtype=np.int16)
        luts = [
            (lut + hue) % 180,
            (lut + sat).clip(0, 255),
            (lut + val).clip(0, 255),
        ]
        return {'lut': np.stack(luts).astype('u1')}

    def apply(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        assert image.ndim == image.shape[-1] == 3

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = super().apply(image, lut)
        return cv2.cvtColor(image.astype('u1'), cv2.COLOR_HSV2RGB)


# ------------------------------- compression -------------------------------


@dataclass
class DegradeJpeg(Transform):
    quality: tuple[int, int] = (0, 15)

    def prepare(self, rg: np.random.Generator, **_) -> dict[str, Any]:
        return {'quality': int(rg.integers(*self.quality))}

    def apply(self, image: np.ndarray, quality: int) -> np.ndarray:
        _, buf = cv2.imencode('.jpg', image,
                              (cv2.IMWRITE_JPEG_QUALITY, quality))
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).reshape(image.shape)


@dataclass
class DegradeQuality(_InlineRandom):
    scale: tuple[float, float] = (0.25, 0.5)
    modes: tuple[str, ...] = ('NEAREST', 'LINEAR', 'INTER_CUBIC', 'AREA')

    def apply(self, image: np.ndarray, rg: np.random.Generator) -> np.ndarray:
        shape = image.shape
        scale = rg.uniform(*self.scale)

        # downscale
        mode = getattr(cv2, f'INTER_{rg.choice(self.modes)}')
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=mode)

        # upscale
        mode = getattr(cv2, f'INTER_{rg.choice(self.modes)}')
        image = cv2.resize(image, shape[1::-1], interpolation=mode)
        return image.reshape(shape)


# ----------------------------- mask alteration -----------------------------


@dataclass
class MaskDropout(_InlineRandom):
    """
    Drops redundant pixels for each class,
    so that (mask == class).sum() <= alpha * mask.size
    """

    alpha: float
    ignore_index: int = -1

    def apply_to_mask(self, mask: np.ndarray,
                      rg: np.random.Generator) -> np.ndarray:
        mask = mask.astype('i8')
        mask, shape = mask.ravel(), mask.shape

        labels_num = int(self.alpha * len(mask))
        for label in np.unique(mask):
            pos, = np.where(mask == label)
            if (to_drop := len(pos) - labels_num) > 0:
                mask[rg.choice(pos, to_drop,
                               replace=False)] = self.ignore_index

        return mask.reshape(*shape)


# --------------------------------- geometry ---------------------------------


@dataclass
class FlipAxis(Transform):
    """
    Flips image/mask vertically/horizontally & rotate by 90 at random.
    In non-isotropic mode (default) flips only horizontally
    """

    isotropic: bool = False

    def prepare(self, rg: np.random.Generator, **_) -> dict[str, Any]:
        ud, lr, rot90 = rg.integers(2, size=3)
        if not self.isotropic:
            ud = rot90 = 0
        return {'ud': ud, 'lr': lr, 'rot90': rot90}

    def apply(self, image: np.ndarray, ud: bool, lr: bool,
              rot90: bool) -> np.ndarray:
        if ud:
            image = image[::-1, :]
        if lr:
            image = image[:, ::-1]
        if rot90:
            image = np.rot90(image, axes=(0, 1))
        return np.ascontiguousarray(image)

    def apply_to_mask(self, mask: np.ndarray, **extra) -> np.ndarray:
        return super().apply(mask)


@dataclass
class WarpAffine(Transform):
    angle: float = 180
    skew: float = .5
    scale: tuple[float, float] = (1., 1.)
    inter: str = 'LINEAR'

    def prepare(self, rg: np.random.Generator, image: np.ndarray,
                **_) -> dict[str, Any]:
        center = np.array(image.shape[:2][::-1]) / 2
        mat = np.hstack([np.eye(2), -center[:, None]])  # move center to origin

        skew = rg.uniform(-self.skew, self.skew)
        skew = [[np.cosh(skew), np.sinh(skew)], [np.sinh(skew), np.cosh(skew)]]

        angle = np.radians(self.angle)
        angle = rg.uniform(-angle, angle)
        rotate = [[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]]

        mat = np.array(rotate) @ np.array(skew) @ mat / rg.uniform(*self.scale)

        mat[:, 2] += center  # restore center
        return {'mat': mat}

    def _apply(self, image: np.ndarray, mat: np.ndarray,
               inter: int) -> np.ndarray:
        return cv2.warpAffine(
            image,
            mat,
            image.shape[:2],
            flags=cv2.WARP_INVERSE_MAP + inter,
            borderMode=cv2.BORDER_REFLECT_101)

    def apply(self, image: np.ndarray, mat: np.ndarray) -> np.ndarray:
        return self._apply(image, mat, getattr(cv2, f'INTER_{self.inter}'))

    def apply_to_mask(self, mask: np.ndarray, mat: np.ndarray) -> np.ndarray:
        return self._apply(mask, mat, cv2.INTER_NEAREST)


@dataclass
class Elastic(Transform):
    """Elastic deformation of image

    Parameters:
    - scale - max offset for each pixel
    - sigma - size of gaussian kernel
    """

    scale: float = 1
    sigma: float = 50
    inter: str = 'LINEAR'

    def prepare(self, rg: np.random.Generator, image: np.ndarray,
                **_) -> dict[str, Any]:
        offsets = rg.random((2, *image.shape[:2]), dtype='f4')
        offsets *= (2 * self.scale)
        offsets -= self.scale

        for dim, (off, size) in enumerate(zip(offsets, image.shape[:2])):
            shape = np.where(np.arange(2) == dim, size, 1)
            off += np.arange(size).reshape(shape)
            cv2.GaussianBlur(off, (17, 17), self.sigma, dst=off)
        return {'offsets': offsets[::-1]}

    def _apply(self, image: np.ndarray, offsets: np.ndarray,
               inter: int) -> np.ndarray:
        return cv2.remap(
            image, *offsets, inter, borderMode=cv2.BORDER_REFLECT_101)

    def apply(self, image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return self._apply(image, offsets, getattr(cv2, f'INTER_{self.inter}'))

    def apply_to_mask(self, mask: np.ndarray,
                      offsets: np.ndarray) -> np.ndarray:
        return self._apply(mask, offsets, cv2.INTER_NEAREST)


@dataclass
class CutOut(_InlineRandom):
    max_holes: int = 80
    size: int = 8
    fill_value: int = 0

    def apply(self, image: np.ndarray, rg: np.random.Generator) -> np.ndarray:
        num_holes = rg.integers(self.max_holes)
        if not num_holes:
            return image

        anchors = rg.integers(0, image.shape[:2], size=(num_holes, 2))

        # [N, dims, (min, max)]
        holes = anchors[:, :, None] + [-self.size // 2, self.size // 2]
        holes = holes.clip(0, np.array(image.shape[:2])[:, None])

        image = image.copy()
        for (y0, y1), (x0, x1) in holes:
            image[y0:y1, x0:x1] = self.fill_value
        return image
