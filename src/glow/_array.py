__all__ = ['aceil', 'afloor', 'apack', 'around', 'smallest_dtype']

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_Scalar_co = TypeVar('_Scalar_co', bound=np.number, covariant=True)


def _force_int(op: np.ufunc, a: npt.NDArray[np.number],
               dtype: np.dtype[_Scalar_co]) -> npt.NDArray[_Scalar_co]:
    assert dtype.kind in 'iu'
    match a.dtype.kind:
        case 'b' | 'u' | 'i':
            return a.astype(dtype)
        case 'f':
            return op(a, out=np.empty_like(a, dtype), casting='unsafe')
        case _:
            raise ValueError(f'Unsupported dtype: {a.dtype}')


def around(x: npt.NDArray[np.number],
           dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.round(x).astype(int)`"""
    return _force_int(np.rint, x, np.dtype(dtype))


def aceil(x: npt.NDArray[np.number],
          dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.ceil(x).astype(int)`"""
    return _force_int(np.ceil, x, np.dtype(dtype))


def afloor(x: npt.NDArray[np.number],
           dtype: npt.DTypeLike = int) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.floor(x).astype(int)`"""
    return _force_int(np.floor, x, np.dtype(dtype))


def smallest_dtype(a_min: float | int,
                   a_max: float | int | None = None) -> np.dtype:
    r = np.min_scalar_type(a_min)
    return r if a_max is None else np.result_type(r, a_max)


def apack(a: npt.ArrayLike,
          a_min: int | None = None,
          a_max: int | None = None) -> npt.NDArray[np.integer]:
    """Convert array to smallest dtype"""
    a = np.asarray(a)
    if not a.size:
        return a
    if a_min is None:
        a_min = a.min()
    if a_max is None:
        a_max = a.max()
    if (dtype := smallest_dtype(a_min, a_max)).itemsize < a.itemsize:
        return a.astype(dtype)
    return a
