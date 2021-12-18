from __future__ import annotations

from types import MethodType

__all__ = ['Pickler', 'Reducer', 'move_to_shmem']

import io
import logging
import mmap
import os
import pickle
import secrets
import sys
import tempfile
import weakref
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copyreg import dispatch_table  # type: ignore
from itertools import count, starmap
from pathlib import Path
from typing import BinaryIO, NamedTuple, TypeVar

import loky
from loky.backend.resource_tracker import maybe_unlink, register

from ._import_hook import when_imported

if sys.platform == 'win32':
    import _winapi

_T = TypeVar('_T')

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

_MIN_OOB_BYTES = 1_000_000
_IPC_REDUCERS: dict[type, Callable] = {}

# ---------------------------------- win32 ----------------------------------


def _win32_get(size: int, name: str | None) -> tuple[mmap.mmap, str]:
    assert sys.platform == 'win32'

    if name is None:  # Create new mmap, never used before
        while True:  # Loop until free name is found
            name = f'wnsm_{secrets.token_hex(4)}'
            hm = _winapi.CreateFileMapping(
                _winapi.INVALID_HANDLE_VALUE,
                _winapi.NULL,
                _winapi.PAGE_READWRITE,
                0xFFFFFFFF & (size >> 32),
                0xFFFFFFFF & size,
                name,
            )
            if _winapi.GetLastError() != _winapi.ERROR_ALREADY_EXISTS:
                break
            _winapi.CloseHandle(hm)

    else:  # Use existing
        hm = _winapi.OpenFileMapping(_winapi.FILE_MAP_READ, False, name)

    mm = mmap.mmap(-1, size, name)  # Bind to name
    _winapi.CloseHandle(hm)
    return mm, name


# ---------------------------------- posix -----------------------------------

_SYSTEM_SHM_MIN_SIZE = int(2e9)
_SYSTEM_SHM = Path('/dev/shm')
_SYSTEM_TEMP = Path(tempfile.gettempdir())


def _get_shm_dir() -> Path:
    if sys.platform != 'win32':
        try:
            stats = os.statvfs(_SYSTEM_SHM)
            if stats.f_bsize * stats.f_bavail > _SYSTEM_SHM_MIN_SIZE:
                return _SYSTEM_SHM
        except OSError:
            pass
    return _SYSTEM_TEMP


_SHM_ROOT = _get_shm_dir()


def _posix_finalize(name: str, fd: int, private: bool = False):
    os.close(fd)
    if private:
        os.unlink(name)
    else:
        maybe_unlink(name, 'file')


def _posix_create(size: int) -> tuple[mmap.mmap, str]:
    assert sys.platform != 'win32'

    while True:
        name = f'{_SHM_ROOT}/psm_{secrets.token_hex(4)}'
        try:
            fd = os.open(name, os.O_RDWR | os.O_EXCL | os.O_CREAT, mode=0o600)
            break
        except FileExistsError:
            pass

    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
    except OSError:
        _posix_finalize(name, fd, private=True)  # Decref, creation failed
        raise
    else:
        register(name, 'file')  # Incref for creation
        weakref.finalize(mm, _posix_finalize, name, fd)  # Decref on __del__
        return mm, name


def _posix_open(size: int, name: str) -> mmap.mmap:
    fd = os.open(name, os.O_RDONLY, mode=0o600)
    try:
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    except OSError:
        _posix_finalize(name, fd)  # Decref, transfer failed
        raise
    else:
        weakref.finalize(mm, _posix_finalize, name, fd)  # Decref on __del__
        return mm


# ------------------------------ shared memory ------------------------------


class MemMap:
    __slots__ = ('_mmap', 'name', 'size')

    @classmethod
    def from_view(cls, view: memoryview) -> MemMap:
        shm = cls(view.nbytes)
        with shm.view as v:
            v[:] = view
        return shm

    def __init__(self, size: int = 0, name: str | None = None) -> None:
        if size <= 0:
            raise ValueError("'size' must be a positive integer")

        self.size = size
        if sys.platform == 'win32':
            self._mmap, self.name = _win32_get(size, name)
        elif name is None:
            self._mmap, self.name = _posix_create(size)
        else:
            self.name = name
            self._mmap = _posix_open(size, name)

    @property
    def uid(self) -> int:
        return id(self._mmap)

    @property
    def view(self) -> memoryview:
        return memoryview(self._mmap)

    def __reduce__(self):
        if sys.platform != 'win32':
            register(self.name, 'file')
        return self.__class__, (self.size, self.name)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.size}, {self.name!r})'


# -------------------------------- reduction --------------------------------


class _Memos(list[MemMap]):
    def collect(self, p: pickle.PickleBuffer) -> bool:
        with p.raw() as v:
            if v.nbytes < _MIN_OOB_BYTES:  # Save as usual
                return True

            self.append(MemMap.from_view(v))
            return False  # Collect as out-of-band data


def _dump(fp: BinaryIO,
          obj: object,
          reducers: Mapping | None = None) -> list[MemMap]:
    memos = _Memos()
    p = pickle.Pickler(fp, -1, buffer_callback=memos.collect)
    p.dispatch_table = ((dispatch_table | _IPC_REDUCERS)  # type: ignore
                        if reducers is None else reducers)
    p.dump(obj)
    return memos


class _PickleProxy(NamedTuple):
    uid: int
    data: bytes
    memos: list[MemMap]


def _reconstruct(data: bytes, *memos: MemMap):
    with ExitStack() as s:
        buffers = (s.enter_context(m.view) for m in memos)
        return pickle.loads(data, buffers=buffers)


class Reducer:
    def __init__(self, pid: int | None = None, is_main: bool = True):
        self.pid = pid or os.getpid()
        self.is_main = is_main
        self.cache: dict[int, MemMap] = {}

    def __reduce__(self) -> tuple:
        return type(self), (self.pid, False)

    def __call__(self, obj: _PickleProxy):
        if sys.platform == 'win32':
            if self.is_main:  # Incref and bind to reducer
                self.cache |= {shm.uid: shm for shm in obj.memos}
        else:
            ...

        return _reconstruct, (obj.data, *obj.memos)


class Pickler(pickle.Pickler):
    def __init__(self, writer: io.BytesIO, protocol=None):
        self.writer = writer
        super().__init__(writer, protocol=-1)  # Override protocol to 5

    def dump(self, obj):
        if _PickleProxy not in self.dispatch_table:
            return super().dump(obj)  # Fallback to basic dump

        if not (memos := _dump(self.writer, obj, self.dispatch_table)):
            return None  # Dump is complete

        obj = _PickleProxy(id(obj), self.writer.getvalue(), memos)
        self.writer.seek(0)
        return super().dump(obj)  # Dump with compact pickle buffer data


@when_imported('torch')
def _torch_hook(torch):
    _IPC_REDUCERS.update({
        torch.Tensor: torch.multiprocessing.reductions.reduce_tensor,
        **{
            t: torch.multiprocessing.reductions.reduce_storage
            for t in torch.storage._StorageBase.__subclasses__()
        },
    })


# --------------------------------- proxies ----------------------------------


class _Chunked(NamedTuple):
    obj: Callable

    def __call__(self, *chunk) -> list:
        return list(starmap(self.obj, chunk))


_SAVED = None
_SAVED_ID = ''
_COUNT = count()
_CACHE: dict[int, tuple[weakref.ref, int]] = {}


def _drop(addr):
    _CACHE.pop(addr, None)


def _is_same(lhs, rhs):
    if not isinstance(lhs, MethodType) or not isinstance(rhs, MethodType):
        return lhs is rhs
    return lhs.__self__ is rhs.__self__ and lhs.__func__ is rhs.__func__


def _get_uid(obj) -> str:
    """Generate always unique id for object"""
    addr = id(obj)
    if ref_n := _CACHE.get(addr):
        ref, n = ref_n
        if _is_same(ref(), obj):
            return f'{addr:x}+{n:x}'

    # Collision or missing reference
    tp = weakref.WeakMethod if isinstance(obj, MethodType) else weakref.ref
    ref = tp(obj, lambda _: _drop(addr))
    n = next(_COUNT)
    _CACHE[addr] = ref, n
    return f'{addr:x}+{n:x}'


class _LazyChunked:
    __slots__ = ('uid', 'base', 'memos')

    def __init__(self, obj):
        s = io.BytesIO()
        self.memos = _dump(s, obj)
        self.base = s.getvalue()
        self.uid = _get_uid(obj)

    def get(self):
        global _SAVED, _SAVED_ID
        if _SAVED is None or _SAVED_ID != self.uid:
            _SAVED_ID = self.uid
            _SAVED = _reconstruct(self.base, *self.memos)

        assert _SAVED is not None
        return _SAVED

    def __call__(self, *chunk) -> list:
        return list(starmap(self.get(), chunk))


def move_to_shmem(fn: Callable[..., _T]) -> Callable[..., list[_T]]:
    return _Chunked(fn)  # type: ignore
    return _LazyChunked(fn)  # type: ignore


loky.set_loky_pickler(__name__)
