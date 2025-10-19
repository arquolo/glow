__all__ = ['move_to_shmem']

import copyreg
import io
import mmap
import os
import pickle
import sys
import tempfile
import weakref
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import loky

from ._import_hook import when_imported
from ._types import Callback

_SYSTEM_SHM_MIN_SIZE = int(2e9)
_SYSTEM_SHM = Path('/dev/shm')
_SYSTEM_TEMP = Path(tempfile.gettempdir())

reducers: dict[type, Callable] = {}
loky.set_loky_pickler('pickle')


def _get_shm_dir() -> Path:
    if sys.platform != 'win32':
        try:
            stats = os.statvfs(_SYSTEM_SHM)
            if stats.f_bsize * stats.f_bavail > _SYSTEM_SHM_MIN_SIZE:
                return _SYSTEM_SHM
        except OSError:
            pass
    return _SYSTEM_TEMP


class _Proxy[*Ts, R]:
    call: Callable[[*Ts], R]

    def __call__(self, *args: *Ts) -> R:
        return self.call(*args)


class _NullProxy[*Ts, R](_Proxy[*Ts, R]):
    __slots__ = ('call',)

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        self.call = call


class _Mmap:
    __slots__ = ('__weakref__', 'buf', 'size', 'tag')
    _shm_root = _get_shm_dir()

    @classmethod
    def from_bytes(cls, data: memoryview, tag: str) -> '_Mmap':
        mv = cls(data.nbytes, f'shm-{os.getpid()}-{tag}', create=True)
        mv.buf[:] = data
        return mv

    def __init__(self, size: int, tag: str, *, create: bool = False) -> None:
        self.size = size
        self.tag = tag
        if create:
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            access = mmap.ACCESS_WRITE
        else:
            flags, access = os.O_RDONLY, mmap.ACCESS_READ

        if sys.platform == 'win32':
            self.buf = mmap.mmap(-1, self.size, self.tag, access=access)
        else:
            name = self._shm_root / f'psm_{self.tag}'
            fd = os.open(name, flags, 0o600)
            # resource_tracker.register(name.as_posix(), 'file')
            if create:
                os.ftruncate(fd, self.size)

            self.buf = mmap.mmap(fd, self.size, access=access)
            if create:
                weakref.finalize(self.buf, os.unlink, name)
            # weakref.finalize(self.buf, resource_tracker.maybe_unlink,
            #                  name.as_posix(), 'file')
            weakref.finalize(self.buf, os.close, fd)

    def __reduce__(self):
        return type(self), (self.size, self.tag)

    def __sizeof__(self) -> int:
        return self.size


@when_imported('torch')
def _torch_hook(torch) -> None:
    reducers.update(
        {torch.Tensor: torch.multiprocessing.reductions.reduce_tensor}
        | dict.fromkeys(
            torch.storage._StorageBase.__subclasses__(),
            torch.multiprocessing.reductions.reduce_storage,
        )
    )


def _dumps(
    obj: object,
    callback: Callback[pickle.PickleBuffer] | None = None,
) -> bytes:
    fp = io.BytesIO()
    p = pickle.Pickler(fp, -1, buffer_callback=callback)
    p.dispatch_table = copyreg.dispatch_table | reducers
    p.dump(obj)
    return fp.getvalue()


def _mmap_reconstruct(data: bytes, memos: list[_Mmap]):
    buffers = [m.buf for m in memos]
    return pickle.loads(data, buffers=buffers)


class _MmapProxy[*Ts, R](_Proxy[*Ts, R]):
    __slots__ = ('call', 'data', 'memos', 'uid')

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        buffers: list[pickle.PickleBuffer] = []
        self.uid = id(call)
        self.call = call
        self.data = _dumps(call, callback=buffers.append)
        self.memos = []
        for i, buf in enumerate(buffers):
            with buf.raw() as m:
                self.memos += [_Mmap.from_bytes(m, f'{self.uid:x}-{i}')]

    def __reduce__(self) -> tuple:
        return _mmap_reconstruct, (self.data, self.memos)


def _shn_reconstruct(data: bytes, memos: list[tuple[SharedMemory, int]]):
    buffers = [sm.buf[:size] for sm, size in memos]
    return pickle.loads(data, buffers=buffers)


class _ShmemProxy[*Ts, R](_Proxy[*Ts, R]):
    __slots__ = ('call', 'data', 'memos')

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        buffers: list[pickle.PickleBuffer] = []
        self.call = call
        self.data = _dumps(call, callback=buffers.append)
        self.memos = []
        for buf in buffers:
            with buf.raw() as m:
                sm = SharedMemory(create=True, size=m.nbytes)
                sm.buf[:] = m
                self.memos.append((sm, sm.size))

    def __reduce__(self) -> tuple:
        return _shn_reconstruct, (self.data, self.memos)


def move_to_shmem[*Ts, R](fn: Callable[[*Ts], R]) -> Callable[[*Ts], R]:
    return _NullProxy(fn)
    return _ShmemProxy(fn)
