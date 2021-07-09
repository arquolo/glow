__all__ = ['affine', 'dither', 'flip', 'grid_shuffle', 'mask_dropout']

from types import MappingProxyType
from typing import Literal

import cv2
import numba
import numpy as np

_MATRICES = MappingProxyType({
    key: cv2.normalize(np.array(mat, 'f4'), None, norm_type=cv2.NORM_L1)
    for key, mat in {
        'jarvis-judice-ninke': [
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ],
        'sierra': [
            [0, 0, 0, 5, 3],
            [2, 4, 5, 4, 2],
            [0, 2, 3, 2, 0],
        ],
        'stucki': [
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
        ],
    }.items()
})
_DitherKind = Literal['jarvis-judice-ninke', 'stucki', 'sierra']


@numba.jit
def _dither(image: np.ndarray, mat: np.ndarray, quant: int) -> np.ndarray:
    for y in range(image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            old = image[y, x]
            new = (old // quant) * quant
            delta = ((old - new) * mat).astype(image.dtype)
            image[y:y + 3, x - 2:x + 3] += delta
            image[y, x] = new

    return image[:-2, 2:-2]


def dither(image: np.ndarray,
           bits: int = 3,
           kind: _DitherKind = 'stucki') -> np.ndarray:
    mat = _MATRICES[kind]
    assert image.dtype == 'u1'
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

    image = _dither(image, mat, quant)
    return image.clip(0, max_value - quant).astype(dtype)  # type: ignore


def grid_shuffle(image: np.ndarray,
                 mask: np.ndarray = None,
                 *,
                 rng: np.random.Generator,
                 grid=4) -> tuple[np.ndarray, ...]:
    axes = (
        np.linspace(0, size, grid + 1, dtype=np.int_)
        for size in image.shape[:2])
    mats = np.stack(np.meshgrid(*axes, indexing='ij'))

    anchors = mats[:, :-1, :-1]
    tiles_sizes = (mats[:, 1:, 1:] - anchors).transpose(1, 2, 0)
    indices = np.indices((grid, grid)).transpose(1, 2, 0)

    for axis in range(2):
        indices = rng.permutation(indices, axis=axis)

    for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
        eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
        indices[eq_mat] = rng.permutation(indices[eq_mat])

    old_pos = anchors[:, indices[..., 0], indices[..., 1]].reshape(2, -1)
    new_pos = anchors.reshape(2, -1)

    sizes = tiles_sizes.reshape(-1, 2)
    tiles = np.stack([new_pos.T, old_pos.T, sizes], axis=1)

    sources = [image]
    if mask is not None:
        sources.append(mask)

    results = []
    for v in sources:
        new_v = np.empty_like(v)
        for (y2, x2), (y1, x1), (ys, xs) in tiles:
            new_v[y2:y2 + ys, x2:x2 + xs] = v[y1:y1 + ys, x1:x1 + xs]
        results.append(new_v)
    return tuple(results)


def affine(image: np.ndarray,
           skew: float,
           angle: float,
           scale: float,
           inter: int = cv2.INTER_LINEAR,
           border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    center = np.array(image.shape[1::-1]) / 2
    mat = np.hstack([np.eye(2), -center[:, None]])  # move center to origin

    angle = np.radians(angle)
    rotate = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]

    shear = [[np.cosh(skew), np.sinh(skew)], [np.sinh(skew), np.cosh(skew)]]

    mat = np.array(rotate) @ np.array(shear) @ mat / scale

    mat[:, 2] += center  # restore center

    flags = inter + cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(
        image, mat, image.shape[:2], flags=flags, borderMode=border)


def flip(image: np.ndarray, ud: bool, lr: bool, rot90: bool) -> np.ndarray:
    if ud:
        image = image[::-1, :]
    if lr:
        image = image[:, ::-1]
    if rot90:
        image = np.rot90(image, axes=(0, 1))
    return np.ascontiguousarray(image)


def mask_dropout(mask: np.ndarray,
                 rng: np.random.Generator,
                 alpha: float,
                 ignore_index: int = -1) -> np.ndarray:
    """
    Keep occurence of each class below alpha.
    Redundant values are replaced with ignore_index.
    """
    dist = np.bincount(mask.ravel()) / mask.size
    prob_keep = alpha / np.maximum(alpha, dist)

    mask = mask.astype('i8')
    mask[rng.random(mask.shape, dtype='f4') >= prob_keep[mask]] = ignore_index
    return mask


def gamma(arr: np.ndarray, y: float) -> np.ndarray:
    """Apply gamma Y to array"""
    assert arr.dtype == 'u1'
    assert y > 0

    lut = np.arange(256, dtype='f4')
    lut /= 255
    lut **= y
    lut *= 255
    lut = lut.round().astype('u1')

    return cv2.LUT(arr.ravel(), lut).reshape(arr.shape)


def gamma_dq(arr: np.ndarray,
             y: float,
             rng: np.random.Generator,
             qbits: int = 7) -> np.ndarray:
    """Apply gamma Y to array with dequantization"""
    assert arr.dtype == 'u1'
    assert y > 0
    assert 0 <= qbits < 8
    peak = 256 << qbits

    lut = np.linspace(0, 1, num=257, dtype='f4')
    lut **= y
    lut *= peak  # Increase precision
    lut = lut.astype('u2')
    lut = lut.clip(0, peak - 1)  # type: ignore  # Remove overshoot

    arr = rng.integers(lut[arr], lut[1:][arr], dtype='u2', endpoint=True)
    arr >>= qbits  # Round to 8 bits
    return arr.astype('u1')
