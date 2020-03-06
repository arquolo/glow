__all__ = ('serialize', )

import copyreg
import ctypes
import io
import logging
import mmap
import os
import random
import sys
import threading
import weakref
from dataclasses import dataclass, field, InitVar
from itertools import starmap
from typing import Any, Callable, ClassVar, Dict, List, NamedTuple, Optional

import loky
import wrapt

if sys.version_info >= (3, 8):
    import pickle
    from multiprocessing.shared_memory import SharedMemory
else:
    import pickle5 as pickle

dispatch_table_patches: Dict[type, Callable] = {}
loky.set_loky_pickler(pickle.__name__)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

_LIBRT = None
if sys.platform != 'win32':
    _LIBRT = ctypes.CDLL('librt.so')


class _Item:
    __slots__ = ('item', 'uid')

    def __init__(self, item):
        self.item = item
        self.uid = id(item)


class _Proxy:
    __slots__ = ('_item', 'uid')

    def __init__(self, item: _Item):
        self._item = item
        self.uid = id(item)

    def get(self) -> _Item:
        return self._item


class _Cached(_Proxy):
    __slots__ = ('_proxy', 'uid')
    _saved: ClassVar[Optional[_Item]] = None

    def __init__(self, proxy: _Proxy):
        self._proxy = proxy
        self.uid = proxy.uid

    def get(self) -> _Item:
        if self._saved is None or self._saved.uid != self.uid:
            self.__class__._saved = self._proxy.get()

        assert self._saved is not None
        return self._saved


class _Mmap:
    __slots__ = ('size', 'tag', 'buf', '__weakref__')

    @classmethod
    def from_bytes(cls, data: memoryview, tag: str) -> '_Mmap':
        mv = cls(data.nbytes, f'shm-{tag}', create=True)
        mv.buf[:] = data
        return mv

    def __init__(self, size, tag, create=False):
        self.size = size
        self.tag = tag
        if create:
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            access = mmap.ACCESS_WRITE
            logger.debug(f'_Mmap.__init__: {self.tag}/{id(self):x}: create')
        else:
            flags, access = os.O_RDONLY, mmap.ACCESS_READ
            logger.debug(f'_Mmap.__init__: {self.tag}/{id(self):x}: open')

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

        weakref.finalize(self, logger.debug,
                         f'_Mmap.__del__ : {self.tag}/{id(self):x}')

    def __reduce__(self):
        return type(self), (self.size, self.tag)

    def __sizeof__(self):
        return self.size


# -------------------------------- untested --------------------------------
_CACHE: Dict[int, Any] = {}
_LOCK = threading.RLock()
# _BOOST = True
_BOOST = False

# from .wrap.cache import memoize


# @memoize(100_000_000, policy='lru', key_fn=id)
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


class _ShmemProxy(_Proxy):
    def __init__(self, item: _Item) -> None:
        buffers: List[pickle.PickleBuffer] = []
        self.uid = item.uid
        self._root = _dumps(item, callback=buffers.append)
        self._memos = []
        for i, buf in enumerate(buffers):
            with buf.raw() as m:
                if sys.version_info < (3, 8):
                    tag = f'{os.getpid()}-{item.uid:x}-{i}'
                    self._memos.append(_Mmap.from_bytes(m, tag))
                else:
                    memo = SharedMemory(create=True, size=m.nbytes)
                    memo.buf[:] = m
                    self._memos.append((memo, memo.size))
        logger.debug(f'_ShmemProxy.__init__: {self.uid}')
        weakref.finalize(self, logger.debug,
                         f'_ShmemProxy.__del__ : {self.uid}')

    def get(self) -> _Item:
        logger.debug(f'_ShmemProxy.get: {self.uid}')
        if sys.version_info < (3, 8):
            buffers = [m.buf for m in self._memos]
        else:
            buffers = [m.buf[:size] for m, size in self._memos]
        return pickle.loads(self._root, buffers=buffers)


class _Task(NamedTuple):
    proxy: _Proxy

    def __call__(self, *chunk):
        return tuple(starmap(self.proxy.get().item, chunk))


def serialize(fn, mp=True) -> _Task:
    item = _Item(fn)
    proxy = _Proxy(item) if (_BOOST or not mp) else _ShmemProxy(item)
    proxy = _Cached(proxy)
    return _Task(proxy)
