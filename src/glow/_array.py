__all__ = ['aceil', 'afloor', 'around']

import numpy as np
import numpy.typing as npt


def around(x: npt.NDArray[np.floating],
           dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.round(x).astype(int)`"""
    dtype = np.dtype(dtype)
    assert x.dtype.kind == 'f'
    assert dtype.kind in 'iu'
    return np.rint(x, out=np.empty_like(x, dtype), casting='unsafe')


def aceil(x: npt.NDArray[np.floating],
          dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.ceil(x).astype(int)`"""
    dtype = np.dtype(dtype)
    assert x.dtype.kind == 'f'
    assert dtype.kind in 'iu'
    return np.ceil(x, out=np.empty_like(x, dtype), casting='unsafe')


def afloor(x: npt.NDArray[np.floating],
           dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.floor(x).astype(int)`"""
    dtype = np.dtype(dtype)
    assert x.dtype.kind == 'f'
    assert dtype.kind in 'iu'
    return np.floor(x, out=np.empty_like(x, dtype), casting='unsafe')
