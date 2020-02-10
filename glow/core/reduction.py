__all__ = ('serialize', )

import copyreg
import io
import mmap
import os
import pathlib
import sys
import tempfile
import weakref
from dataclasses import dataclass, field, InitVar
from itertools import starmap
from typing import (Callable, ClassVar, Dict, Hashable, List, NamedTuple,
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
    def from_bytes(cls, data: bytes, tag: str) -> '_Mmap':
        mv = cls(len(data), f'shm-{tag}', create=True)
        mv.buf[:] = data
        return mv

    def __post_init__(self, create):
        access = mmap.ACCESS_WRITE if create else mmap.ACCESS_READ

        if sys.platform == 'win32':
            args = (-1, self.size, self.tag)
        else:
            args = self._posix_args(create)

        self.buf = mmap.mmap(*args, access=access)
        weakref.finalize(self, self.buf.close)

    def _posix_args(self, create):
        filepath = pathlib.Path(tempfile.gettempdir(), self.tag)
        if create:
            fd = os.open(filepath, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.ftruncate(fd, self.size)
            weakref.finalize(self, filepath.unlink)
        else:
            fd = os.open(filepath, os.O_RDONLY)

        weakref.finalize(self, os.close, fd)
        return (fd, self.size)

    def __reduce__(self):
        return type(self), (self.size, self.tag)


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
            memo = SharedMemory(create=True, size=len(buf.raw()))
            memo.buf[:] = buf.raw()
            self.memos.append((memo, memo.size))

    def load(self):
        buffers = [s.buf[:size] for s, size in self.memos]
        return pickle.loads(self.root, buffers=buffers)


def serialize(fn, mp=True) -> _CallableMixin:
    if not mp:
        return _SimpleProxy(fn)

    if sys.version_info >= (3, 8):
        return _SharedPickleProxy(fn)

    return _MmapProxy(fn)
