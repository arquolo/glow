__all__ = ('serialize', )

import collections
import mmap
import os
import pathlib
import sys
import tempfile
import weakref
from itertools import starmap
from typing import Optional

import loky

from ..decos import Reusable


if sys.version_info >= (3, 8):
    import pickle
    from multiprocessing.shared_memory import SharedMemory
    loky.set_loky_pickler('pickle')
else:
    import pickle5 as pickle
    loky.set_loky_pickler('pickle5')

_Item = collections.namedtuple('_Item', ['item', 'id'])

_GC_TIMEOUT = 10


def _get_dict():
    return loky.backend.get_context().Manager().dict()


class _CallableMixin:
    def __call__(self, *chunk):
        return tuple(starmap(self.item, chunk))


class _SimpleProxy(_CallableMixin):
    def __init__(self, item):
        self.item = item


class _RemoteCacheMixin:
    saved: Optional[_Item] = None

    def __init__(self, id_):
        self.id = id_

    def load(self):
        raise NotImplementedError

    @property
    def item(self):
        if self.saved is None or self.saved.id != self.id:
            type(self).saved = _Item(self.load(), self.id)
        assert self.saved is not None
        return self.saved.item


class _ManagedProxy(_RemoteCacheMixin, _CallableMixin):
    """Uses manager-process. Slowest one"""
    manager: Optional[Reusable] = None

    def __init__(self, item):
        super().__init__(id(item))
        if self.manager is None:
            type(self).manager = Reusable(_get_dict, timeout=_GC_TIMEOUT)
        assert self.manager is not None
        self.shared = self.manager.get()
        self.shared[id(item)] = item

    def load(self):
        return self.shared[self.id]


class _MmapProxyBase(_RemoteCacheMixin, _CallableMixin):
    _access = mmap.ACCESS_READ

    def __init__(self, size, id_, create=False):
        super().__init__(id_)
        if sys.platform == 'win32':
            self.buf = self._buf_win(size, id_, create=create)
        else:
            self.buf = self._buf_posix(size, id_, create=create)
        weakref.finalize(self, self.buf.close)

    def _buf_win(self, size, id_, create=False):
        if create:
            self._access = mmap.ACCESS_WRITE
        return mmap.mmap(-1, size, f'shm-{id_}', access=self._access)

    def _buf_posix(self, size, id_, create=False):
        filepath = pathlib.Path(tempfile.gettempdir(), f'shm-{id_}')
        if create:
            self._access = mmap.ACCESS_WRITE

            fd = os.open(filepath, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.ftruncate(fd, size)
            weakref.finalize(self, filepath.unlink)
        else:
            fd = os.open(filepath, os.O_RDONLY)

        weakref.finalize(self, os.close, fd)
        return mmap.mmap(fd, size, access=self._access)

    def __reduce__(self):
        return _MmapProxyBase, (self.buf.size(), self.id)

    def load(self):
        return pickle.loads(self.buf)


class _MmapProxy(_MmapProxyBase):
    """Fallback for sharedmemory for Python<3.8"""
    def __init__(self, item):
        data = pickle.dumps(item, -1)
        super().__init__(len(data), f'{os.getpid()}-{id(item):x}', create=True)
        self.buf[:] = data


class _SharedPickleProxy(_CallableMixin):
    """Uses sharedmemory. Available on Python 3.8+"""
    def __init__(self, item):
        assert sys.version_info >= (3, 8)
        buffers = []
        self.root = pickle.dumps(item, -1, buffer_callback=buffers.append)
        self.memos = []
        for buf in buffers:
            memo = SharedMemory(create=True, size=len(buf.raw()))
            memo.buf[:] = buf.raw()
            self.memos.append((memo, memo.size))

    @property
    def item(self):
        buffers = [s.buf[:size] for s, size in self.memos]
        return pickle.loads(self.root, buffers=buffers)


def serialize(fn, mp=True) -> _CallableMixin:
    if not mp:
        return _SimpleProxy(fn)

    if sys.version_info >= (3, 8):
        return _SharedPickleProxy(fn)

    return _MmapProxy(fn)
    # return _ManagedProxy(fn)
