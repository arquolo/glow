__all__ = ['imhash_hist']

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL.Image import Image

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
