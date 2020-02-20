__all__ = ('serialize', )

import copyreg
import ctypes
import io
import mmap
import os
import random
import sys
import threading
import weakref
from dataclasses import dataclass, field, InitVar
from itertools import starmap
from typing import (Any, Callable, ClassVar, Dict, Hashable, List, NamedTuple,
                    Optional)

import loky
import wrapt

if sys.version_info >= (3, 8):
    import pickle
    from multiprocessing.shared_memory import SharedMemory
else:
    import pickle5 as pickle

GC_TIMEOUT = 10
dispatch_table_patches: Dict[type, Callable] = {}
loky.set_loky_pickler(pickle.__name__)

_LIBRT = None
if sys.platform != 'win32':
    _LIBRT = ctypes.CDLL('librt.so')


class _Item(NamedTuple):
    item: Callable
    uid: Hashable


class _CallableMixin:
    def load(self) -> Callable:
        ...

    def __call__(self, *chunk):
        return tuple(starmap(self.load(), chunk))


class _SimpleProxy(_CallableMixin):
    def __init__(self, item):
        self._item = item

    def load(self):
        return self._item


@dataclass
class _RemoteCacheMixin:
    uid: Hashable
    saved: ClassVar[Optional[_Item]] = None

    def _load(self) -> Callable:
        ...

    def load(self):
        if self.saved is None or self.saved.uid != self.uid:
            self.__class__.saved = _Item(self._load(), self.uid)

        assert self.saved is not None
        return self.saved.item


@dataclass
class _Mmap:
    size: int
    tag: str
    create: InitVar[bool] = False
    buf: mmap.mmap = field(init=False)

    @classmethod
    def from_bytes(cls, data: memoryview, tag: str) -> '_Mmap':
        tag = f'{tag}-{random.getrandbits(96):x}'
        mv = cls(data.nbytes, f'shm-{tag}', create=True)
        mv.buf[:] = data
        # if __debug__:
        #     print(f'{id(mv):x}: {mv.tag}: create')
        return mv

    def __post_init__(self, create):
        if create:
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            access = mmap.ACCESS_WRITE
        else:
            flags, access = os.O_RDONLY, mmap.ACCESS_READ

        # if __debug__ and not create:
        #     print(f'{id(self):x}: {self.tag}: open')

        if sys.platform == 'win32':
            self.buf = mmap.mmap(-1, self.size, self.tag, access=access)
        else:
            name = f'/psm_{self.tag}'.encode()
            fd = _LIBRT.shm_open(name, flags, 0o600)
            if create:
                os.ftruncate(fd, self.size)

            self.buf = mmap.mmap(fd, self.size, access=access)
            weakref.finalize(self.buf, _LIBRT.shm_unlink, name)
            weakref.finalize(self.buf, os.close, fd)

    # def __del__(self):
    #     if __debug__:
    #         print(f'{id(self):x}: destroyed')

    def __reduce__(self):
        return type(self), (self.size, self.tag)

    def __sizeof__(self):
        return self.size


# -------------------------------- untested --------------------------------
_CACHE: Dict[int, Any] = {}
_LOCK = threading.RLock()
# _BOOST = True
_BOOST = False

from .wrap.cache import memoize


@memoize(100_000_000, policy='lru', key_fn=id)
def _np_reduce(arr):
    uid = id(arr)
    with memoryview(arr) as m:
        memo = _Mmap.from_bytes(m, tag=f'{os.getpid()}-{uid:x}')
        return _np_recreate, (uid, memo, m.format, m.shape)


def _np_reduce_cached(arr):
    uid = id(arr)

    with _LOCK:
        args = _CACHE.get(uid)
        if args is not None:
            return args
        args = _CACHE[uid] = _np_reduce(arr)

    def finalize():
        with _LOCK:
            _CACHE.pop(uid)

    weakref.finalize(arr, finalize)
    return args  # noqa: R504


def _np_recreate(uid, memo, fmt, shape):
    import numpy as np
    return np.asarray(memoryview(memo.buf).cast(fmt, shape))  # type: ignore


if _BOOST:

    @wrapt.when_imported('numpy')
    def _(numpy):
        dispatch_table_patches[numpy.ndarray] = _np_reduce
        # dispatch_table_patches[numpy.ndarray] = _np_reduce_cached


# -------------------------------- untested --------------------------------


@wrapt.when_imported('torch')
def _(torch):
    dispatch_table_patches.update({
        torch.Tensor: torch.multiprocessing.reductions.reduce_tensor,
        **{
            t: torch.multiprocessing.reductions.reduce_storage
            for t in torch.storage._StorageBase.__subclasses__()
        }
    })


def _dumps(obj: object,
           callback: Callable[[pickle.PickleBuffer], object] = None) -> bytes:
    fp = io.BytesIO()
    p = pickle.Pickler(fp, -1, buffer_callback=callback)
    p.dispatch_table = copyreg.dispatch_table.copy()  # type: ignore
    p.dispatch_table.update(dispatch_table_patches)
    p.dump(obj)
    return fp.getvalue()


class _MmapProxy(_RemoteCacheMixin, _CallableMixin):
    """Fallback for sharedmemory for Python<3.8"""
    def __init__(self, item):
        super().__init__(id(item))

        buffers: List[pickle.PickleBuffer] = []
        self.root = _dumps(item, callback=buffers.append)
        self.memos = [
            _Mmap.from_bytes(buf.raw(), f'{os.getpid()}-{self.uid:x}-{i}')
            for i, buf in enumerate(buffers)
        ]

    def _load(self):
        buffers = [m.buf[:m.size] for m in self.memos]
        return pickle.loads(self.root, buffers=buffers)


class _SharedPickleProxy(_CallableMixin):
    """Uses sharedmemory. Available on Python 3.8+"""
    def __init__(self, item):
        assert sys.version_info >= (3, 8)
        buffers = []
        self.root = _dumps(item, callback=buffers.append)
        self.memos = []
        for buf in buffers:
            with buf.raw() as m:
                memo = SharedMemory(create=True, size=m.nbytes)
                memo.buf[:] = m
            self.memos.append((memo, memo.size))

    def load(self):
        buffers = [s.buf[:size] for s, size in self.memos]
        return pickle.loads(self.root, buffers=buffers)


def serialize(fn, mp=True) -> _CallableMixin:
    if not mp:
        return _SimpleProxy(fn)

    if _BOOST:
        return _SimpleProxy(fn)

    if sys.version_info >= (3, 8):
        return _SharedPickleProxy(fn)
    return _MmapProxy(fn)
