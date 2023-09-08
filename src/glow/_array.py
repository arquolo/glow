__all__ = ['aceil', 'afloor', 'around']

import numpy as np
import numpy.typing as npt


def around(x: np.ndarray, dtype: npt.DTypeLike = int) -> np.ndarray:
    """Faster alternative to `np.round(x).astype(int)`"""
    return np.rint(x, out=np.empty_like(x, dtype), casting='unsafe')


def aceil(x: np.ndarray, dtype: npt.DTypeLike = int) -> np.ndarray:
    """Faster alternative to `np.ceil(x).astype(int)`"""
    return np.ceil(x, out=np.empty_like(x, dtype), casting='unsafe')


def afloor(x: np.ndarray, dtype: npt.DTypeLike = int) -> np.ndarray:
    """Faster alternative to `np.floor(x).astype(int)`"""
    return np.floor(x, out=np.empty_like(x, dtype), casting='unsafe')
