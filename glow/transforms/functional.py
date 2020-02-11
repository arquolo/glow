__all__ = (
    'bit_noise',
    'cutout',
    'dither',
    'elastic',
    'Compose',
    'Transform',
)

from types import MappingProxyType
from typing import Tuple

import cv2
import numba
import numpy as np
from typing_extensions import Literal, Protocol

_MATRICES = MappingProxyType({
    key: cv2.normalize(np.float32(mat), None, norm_type=cv2.NORM_L1)
    for key, mat in {
        'jarvis-judice-ninke': [
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ],
        'sierra': [[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]],
        'stucki': [[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]],
    }.items()
})
_DitherKind = Literal['jarvis-judice-ninke', 'stucki', 'sierra']


class Transform(Protocol):
    def __call__(self, *args: np.ndarray,
                 rg: np.random.Generator) -> Tuple[np.ndarray, ...]:
        ...


class Compose:
    def __init__(self, *callables: Tuple[float, Transform]):
        self.callables = callables

    def __call__(self, *args: np.ndarray,
                 rg: np.random.Generator) -> Tuple[np.ndarray, ...]:
        for prob, fn in self.callables:
            if rg.uniform() <= prob:
                new_args = fn(*args, rg=rg)
                args = new_args + args[len(new_args):]

        return args


@numba.jit
def dither(image: np.ndarray,
           *_,
           rg: np.random.Generator = None,
           bits: int = 3,
           kind: _DitherKind = 'stucki') -> Tuple[np.ndarray, ...]:
    mat = _MATRICES[kind]
    if image.ndim == 3:
        mat = mat[..., None]
        channel_pad = [(0, 0)]
    else:
        channel_pad = []
    image = np.pad(image, [(0, 2), (2, 2)] + channel_pad, mode='constant')

    dtype = image.dtype
    if dtype == 'uint8':
        max_value = 256
        image = image.astype('i2')
    else:
        max_value = 1
    quant = max_value / 2 ** bits

    for y in range(image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            old = image[y, x]
            new = np.floor(old / quant) * quant
            delta = ((old - new) * mat).astype(image.dtype)
            image[y:y + 3, x - 2:x + 3] += delta
            image[y, x] = new

    image = image[:-2, 2:-2]
    return image.clip(0, max_value - quant).astype(dtype),


def bit_noise(image: np.ndarray,
              *_,
              rg: np.random.Generator,
              keep: int = 4,
              count: int = 8) -> Tuple[np.ndarray]:
    residual = image.copy()
    out = np.zeros_like(image)
    for n in range(1, 1 + count):
        thres = 0.5 ** n
        plane = (residual >= thres).astype(residual.dtype) * thres
        if n <= keep:
            out += plane
        else:
            out += rg.choice([0, thres], size=image.shape)
        residual -= plane
    return out,


def elastic(image: np.ndarray,
            mask: np.ndarray,
            *_,
            rg: np.random.Generator,
            scale: float = 1,
            sigma: float = 50,
            interp=cv2.INTER_LINEAR,
            border=cv2.BORDER_REFLECT_101) -> Tuple[np.ndarray, ...]:
    """Elastic deformation of image

    Parameters:
      - `scale` - max offset for each pixel
      - `sigma` - size of gaussian kernel
    """
    offsets = rg.random((2, *image.shape[:2]), dtype='f4')
    offsets *= (2 * scale)
    offsets -= scale

    for dim, (off, size) in enumerate(zip(offsets, image.shape[:2])):
        shape = np.where(np.arange(2) == dim, size, 1)
        off += np.arange(size).reshape(shape)
        cv2.GaussianBlur(off, (17, 17), sigma, dst=off)

    return tuple(
        cv2.remap(m, *offsets[::-1], interp_, borderMode=border)
        for m, interp_ in ((image, interp), (mask, cv2.INTER_NEAREST)))


def cutout(image: np.ndarray,
           *_,
           rg: np.random.Generator,
           max_holes=80,
           size=8,
           fill_value=0) -> Tuple[np.ndarray]:
    num_holes = rg.integers(max_holes)
    if not num_holes:
        return image,

    anchors = rg.integers(0, image.shape[:2], size=(num_holes, 2))

    # [N, dims, (min, max)]
    holes = anchors[:, :, None] + [-size // 2, size // 2]
    holes = holes.clip(0, np.asarray(image.shape[:2])[:, None])

    image = image.copy()
    for (y0, y1), (x0, x1) in holes:
        image[y0:y1, x0:x1] = fill_value
    return image,
