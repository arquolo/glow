from __future__ import annotations  # until 3.10

__all__ = ['dither', 'grid_shuffle']

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
           *_,
           rg: np.random.Generator = None,
           bits: int = 3,
           kind: _DitherKind = 'stucki') -> tuple[np.ndarray, ...]:
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
    return image.clip(0, max_value - quant).astype(dtype),


def grid_shuffle(image: np.ndarray,
                 mask: np.ndarray = None,
                 *,
                 rg: np.random.Generator,
                 grid=4) -> tuple[np.ndarray, ...]:
    axes = (
        np.linspace(0, size, grid + 1, dtype=np.int_)
        for size in image.shape[:2])
    mats = np.stack(np.meshgrid(*axes, indexing='ij'))

    anchors = mats[:, :-1, :-1]
    tiles_sizes = (mats[:, 1:, 1:] - anchors).transpose(1, 2, 0)
    indices = np.indices((grid, grid)).transpose(1, 2, 0)

    for axis in range(2):
        indices = rg.permutation(indices, axis=axis)

    for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
        eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
        indices[eq_mat] = rg.permutation(indices[eq_mat])

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
    return *results,
