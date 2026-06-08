__all__ = [
    'circle',
    'imhash_hist',
    'imresize_categorical',
    'imresize_multichannel',
    'imrotate',
]

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL.Image import Image

_BLOCK_SIZE = 4

type _U8 = npt.NDArray[np.uint8]
type _F32 = npt.NDArray[np.float32]
type _AnyImage = Path | str | Image | _F32 | _U8


def imhash_hist(x: _AnyImage, /, *, bins: int = 4) -> _F32 | None:
    """Compute histogram-based L2-normalized image hash.

    To measure distance use dot product (i.e. `hash1 @ hash2`).
    Large dot product means images have similar color distribution.
    """
    image = _ensure_image(x)
    if image is None:
        return None

    # By default 4x4x4, i.e. 64 values
    hist = cv2.calcHist(
        [image],
        channels=[0, 1, 2],
        mask=None,
        histSize=[bins, bins, bins],
        ranges=[0, 256, 0, 256, 0, 256],
    )
    hist = np.float32(hist).ravel()
    l2 = cv2.norm(hist, normType=cv2.NORM_L2)
    return hist / l2


def _ensure_image(i: _AnyImage, /) -> _U8 | None:
    match i:
        case Path() | str():
            return cv2.imread(str(i), cv2.IMREAD_COLOR)  # type: ignore
        case Image():
            return np.asarray(i.convert('RGB'), 'B')
        case np.ndarray():
            i = _ensure_u8(i)
            return _ensure_rgb(i)
        case _:
            return None


def _ensure_rgb(image: _U8, /) -> _U8:
    match image.shape:
        case (_, _) | (_, _, 1):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # type: ignore
        case (_, _, 3):
            return image
        case _ as unsupported:
            msg = f'Input is not image. Got shape {unsupported}'
            raise ValueError(msg)


def _ensure_u8(image: _F32 | _U8, /) -> _U8:
    dt = image.dtype
    if dt == 'B':
        return image  # type: ignore
    if dt == 'f':
        return (image * 255).clip(0, 255).astype('B')  # assume [0 .. 1]
    msg = f'Unsupported image dtype. Got {dt}'
    raise NotImplementedError(msg)


def imresize_multichannel(
    img: np.ndarray,
    h: int,
    w: int,
    *,
    interpolation: int,
    blksize: int = 4,
) -> np.ndarray:
    """Resize multichannel image.

    Parameters:
    - img - (h w) image of integers
    - h & w - target size
    - interpolation - cv2.INTER... value
    - blksize - N channels to use for block resize
    """
    assert img.ndim == 3
    c = img.shape[-1]
    ret = np.empty((h, w, c), img.dtype)
    for i in range(0, c, blksize):
        cv2.resize(
            img[..., i : i + blksize],
            (w, h),
            dst=ret[..., i : i + blksize],
            interpolation=interpolation,
        )
    return ret


def imresize_categorical(
    img: np.ndarray,
    h: int,
    w: int,
    *,
    interpolation: int = cv2.INTER_CUBIC,
    fill_value: int = 0,
    blksize: int = _BLOCK_SIZE,
) -> np.ndarray:
    """Resize categorical data through one hot.

    Parameters:
    - img - (h w) image of integers
    - h & w - target size
    - fill_value - value to use in case of 0-sized input
    - blksize - N channels to use for block resize
    """
    if img.ndim != 2 and (img.ndim != 3 or img.shape[-1] != 1):
        raise ValueError(f'Only single channel is allowed. Got {img.shape=}')

    u = np.unique(img)
    if u.size <= 1:  # Empty or unary
        if u.size:
            fill_value = u[0]
        return np.full((h, w), fill_value, dtype=img.dtype)

    if u.size == 2:  # Binary
        ret = np.empty((h, w), dtype='B')
        cv2.resize(
            np.where(img == u[1], np.uint8(255), np.uint8(0)),
            (w, h),
            dst=ret,
            interpolation=interpolation,
        )
        return np.where(ret >= 128, u[1], u[0])

    onehot = img[:, :, None] == u[None, None, :]
    onehot = np.where(onehot, np.uint8(255), np.uint8(0))
    onehot = imresize_multichannel(
        onehot, h, w, interpolation=interpolation, blksize=blksize
    )
    return u[onehot.argmax(-1)]


def imrotate[T: (np.float32, np.uint8)](
    img: npt.NDArray[T],
    degrees: float,
    fit: bool | tuple[int, int] = False,
    interpolation: int = cv2.INTER_CUBIC,
    border: int = cv2.BORDER_REPLICATE,
    blksize: int = _BLOCK_SIZE,
) -> npt.NDArray[T]:
    """Rotate image around its center"""
    h, w, *cs = img.shape
    img = img.reshape(h, w, -1)

    radians = np.radians(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    acos, asin = abs(cos), abs(sin)

    if fit is True:
        h2, w2 = int(h * acos + w * asin), int(h * asin + w * acos)
    elif fit:
        h2, w2 = fit
    else:
        h2, w2 = h, w

    rot = np.array([[cos, -sin], [sin, cos]])
    bias = [w - 1, h - 1] - rot @ [w2 - 1, h2 - 1]

    c = img.shape[-1]
    ret = np.empty((h2, w2, c), img.dtype)
    for i in range(0, c, blksize):
        cv2.warpAffine(
            img[:, :, i : i + blksize],
            np.c_[rot, bias / 2],
            (w2, h2),
            dst=ret[:, :, i : i + blksize],
            flags=cv2.WARP_INVERSE_MAP | interpolation,
            borderMode=border,
        )
    return ret.reshape(h2, w2, *cs)


def circle(diameter: int = 45, gain: float = 2) -> npt.NDArray[np.float32]:
    assert diameter >= 4
    assert gain >= 0
    axis = np.linspace(-1, 1, diameter, dtype='f')
    img = axis[:, None] ** 2 + axis[None, :] ** 2
    img[img >= 1] = 0
    if gain != 2:
        img **= gain / 2
    return img
