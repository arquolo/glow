__all__ = ['ArrayForwardReducer', 'reduce_array_backward']

import errno
import os
import stat
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from mmap import mmap
from pickle import dumps, loads
from uuid import uuid4

import numpy as np
from loky.backend import resource_tracker
from numpy.lib.stride_tricks import as_strided

JOBLIB_MMAPS = set()


class _WeakKeyMap(dict):
    def get(self, obj):
        key = id(obj)

        if (rec := super().get(key)) is not None:
            ref, val = rec
            if ref() is obj:
                return val  # noqa: R504

        val = (f'{os.getpid()}-{id(threading.current_thread()):x}-'
               f'{uuid4().hex}.pkl')
        ref = weakref.ref(obj, lambda _: self.pop(key))
        self[key] = ref, val
        return val


def _get_backing_memmap(a):
    while True:
        if (m := getattr(a, 'base', None)) is not None:
            if isinstance(m, mmap):
                return a
            a = m
        else:
            return None


def _rebuild_memmap(filename, dtype, offset, order, shape, strides,
                    total_buffer_len):
    base = np.memmap(
        filename,
        dtype=dtype,
        mode='r',
        offset=offset,
        shape=total_buffer_len,
        order=order)
    if strides is None:
        return base
    return as_strided(base, shape=shape, strides=strides)


def _reduce_memmap_backed(a, m):
    a_start, a_end = np.byte_bounds(a)
    m_start, _ = np.byte_bounds(m)

    offset = (a_start - m_start) + m.offset
    order = 'F' if m.flags.f_contiguous else 'C'

    strides, total_buffer_len = ((None, a.shape) if a.flags.forc else
                                 (a.strides, (a_end - a_start) // a.itemsize))

    return _rebuild_memmap, (m.filename, a.dtype, offset, order, a.shape,
                             strides, total_buffer_len)


def _rebuild_array(subclass, shape, order, dtype, path, unlink_on_gc):
    obj = np.memmap(path, dtype=dtype, mode='r', shape=shape, order=order)

    if (hasattr(obj, '__array_prepare__') and
            subclass not in (np.ndarray, np.memmap)):
        empty = np.core.multiarray._reconstruct(subclass, (0, ), 'b')
        obj = empty.__array_prepare__(obj)

    JOBLIB_MMAPS.add(obj.filename)
    if unlink_on_gc:
        weakref.finalize(obj, resource_tracker.maybe_unlink,
                         obj.filename.as_posix(), 'file')
    return obj


@dataclass(eq=False)
class ArrayForwardReducer:
    _max_nbytes: int
    _unlink_on_gc: bool
    _temp_resolver: Callable = None  # type: ignore
    _memmaped_arrays: _WeakKeyMap = field(default_factory=_WeakKeyMap)
    _memmaped_filenames: set = field(default_factory=set)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self._max_nbytes == other._max_nbytes)

    def __reduce__(self):
        return type(self), (self._max_nbytes, self._unlink_on_gc)

    def __call__(self, a):
        if isinstance((mm := _get_backing_memmap(a)), np.memmap):
            return _reduce_memmap_backed(a, mm)

        if (a.dtype.hasobject or self._max_nbytes is None or
                a.nbytes <= self._max_nbytes):
            return loads, (dumps(a, protocol=-1), )

        temp_folder = self._temp_resolver()
        try:
            temp_folder.mkdir(parents=True)
            temp_folder.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

        path = temp_folder / self._memmaped_arrays.get(a)
        if path not in self._memmaped_filenames:
            self._memmaped_filenames.add(path)
            resource_tracker.register(path.as_posix(), 'file')

        if self._unlink_on_gc:
            resource_tracker.register(path.as_posix(), 'file')

        if isinstance(a, np.memmap):
            a = np.asanyarray(a)

        order = ('F' if a.flags.fnc else 'C')
        if not path.exists():
            with path.open('wb') as fp:
                for chunk in np.nditer(
                        a,
                        flags=['external_loop', 'buffered', 'zerosize_ok'],
                        buffersize=max(16 * 1024 ** 2 // a.itemsize, 1),
                        order=order):
                    fp.write(chunk.tobytes('C'))
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        return _rebuild_array, (type(a), a.shape, order, a.dtype, path,
                                self._unlink_on_gc)


def reduce_array_backward(a):
    m = _get_backing_memmap(a)
    if isinstance(m, np.memmap) and m.filename not in JOBLIB_MMAPS:
        return _reduce_memmap_backed(a, m)
    return loads, (dumps(np.asarray(a), protocol=-1), )
