__all__ = ['aceil', 'afloor', 'apack', 'around', 'pascal', 'smallest_dtype']

import numpy as np
import numpy.typing as npt


def _force_int(
    op: np.ufunc, a: npt.NDArray[np.number], dtype: npt.DTypeLike
) -> npt.NDArray[np.integer]:
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        msg = f'Cannot cast to non-integer dtype: {dtype}'
        raise ValueError(msg)

    match a.dtype.kind:
        case 'b' | 'u' | 'i':
            return a.astype(dtype)
        case 'f':
            return op(a, out=np.empty_like(a, dtype), casting='unsafe')
        case _:
            msg = f'Unsupported dtype: {a.dtype}'
            raise ValueError(msg)


def around(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = int
) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.round(x).astype(int)`."""
    return _force_int(np.rint, x, dtype)


def aceil(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = int
) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.ceil(x).astype(int)`."""
    return _force_int(np.ceil, x, dtype)


def afloor(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = int
) -> npt.NDArray[np.integer]:
    """Faster alternative to `np.floor(x).astype(int)`."""
    return _force_int(np.floor, x, dtype)


def smallest_dtype(
    a_min: float | int, a_max: float | int | None = None
) -> np.dtype:
    r = np.min_scalar_type(a_min)
    return r if a_max is None else np.result_type(r, a_max)


def apack(
    a: npt.ArrayLike | npt.NDArray[np.integer],
    a_min: int | None = None,
    a_max: int | None = None,
) -> npt.NDArray[np.integer]:
    """Convert integer array to smallest dtype."""
    a = np.asarray(a)
    if a.dtype.kind != 'i':
        msg = f'Cannot pack non-integer array: {a.dtype}'
        raise ValueError(msg)
    if not a.size:
        return a

    if a_min is None:
        a_min = a.min()
        assert a_min is not None

    if a_max is None:
        a_max = a.max()

    if (dtype := smallest_dtype(a_min, a_max)).itemsize < a.itemsize:
        return a.astype(dtype)
    return a


def pascal(n: int) -> npt.NDArray[np.int64]:
    if n < 1 or n > 67:
        msg = '`n` must in 1..67 range'
        raise ValueError(msg)
    a = np.zeros(n, 'q')
    a[0] = 1
    for _ in range(n - 1):
        a[1:] += a[:-1]
    return a
