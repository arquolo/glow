__all__ = ['move_to_shmem']

import copyreg
import io
import mmap
import os
import pickle
import sys
import tempfile
import uuid
import weakref
from collections.abc import Callable
from concurrent.futures import Future
from functools import update_wrapper
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Lock
from typing import Literal, Self

import loky

from ._import_hook import register_post_import_hook
from ._types import MpMode

# --------------------------------- factory ----------------------------------


def move_to_shmem[*Ts, R](
    fn: Callable[[*Ts], R], mode: MpMode = True
) -> '_Proxy[*Ts, R]':
    match mode:
        case True:
            return _NullProxy(fn)
        case 'mmap':
            return _MmapParentProxy(fn)
        case 'shmem':
            return _ShmemParentProxy(fn)
        case _:
            raise ValueError(f'Unknown `mp_mode`: {mode}')


# ---------------------------------- proxies ----------------------------------


class _Proxy[*Ts, R]:
    __slots__ = ('__weakref__',)

    def __call__(self, *args: *Ts) -> R:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def wrap_func[**P, F: Future](self, fn: Callable[P, F]) -> Callable[P, F]:
        return fn

    def __exit__(self, *args) -> None:
        return


class _NullProxy[*Ts, R](_Proxy[*Ts, R]):
    __slots__ = ('call',)

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        self.call = call

    def __call__(self, *args: *Ts) -> R:
        return self.call(*args)


class _ChildProxy[*Ts, R]:
    __slots__ = ('call', 'handles')

    def __init__(self, call: Callable[[*Ts], R], *handles: object) -> None:
        self.call = call
        self.handles = handles

    def __call__(self, *args: *Ts) -> R:
        return self.call(*args)

    def __reduce__(self) -> tuple:
        raise RuntimeError('Cannot pass callable back to parent process')


class _ParentProxy[*Ts, R](_Proxy[*Ts, R]):
    __slots__ = ('_lock', '_pending', '_state')

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending = 0
        self._state: Literal['running', 'closing', 'closed'] = 'running'

    def __call__(self, *args: *Ts) -> R:
        raise RuntimeError('Calling moved callable from parent process')

    def wrap_func[**P, F: Future](self, fn: Callable[P, F]) -> Callable[P, F]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> F:
            with self._lock:
                if self._state != 'running':
                    raise RuntimeError('Cannot schedule to closed')
                self._pending += 1
            try:
                fut = fn(*args, **kwargs)
            except BaseException:
                self._done()
                raise
            else:
                fut.add_done_callback(self._done)
                return fut

        return update_wrapper(inner, fn)

    def _done(self, *_) -> None:
        with self._lock:
            self._pending -= 1
            if self._state != 'closing' or self._pending:
                return
            self._state = 'closed'
        self._close()

    def __exit__(self, *_) -> None:
        with self._lock:
            if self._state == 'closed':
                return
            if self._pending:
                self._state = 'closing'
                return
            self._state = 'closed'
        self._close()

    def _close(self) -> None:
        raise NotImplementedError


class _MmapParentProxy[*Ts, R](_ParentProxy[*Ts, R]):
    __slots__ = ('data', 'memos')

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        super().__init__()
        uid = uuid.uuid4().int
        self.data, buffers = _dumps5(call)
        self.memos: list[_Mmap] = []
        for i, buf in enumerate(buffers):
            with buf.raw() as m:
                self.memos += [_new_mmap(m, f'{uid:x}-{i}')]

    def __reduce__(self) -> tuple:
        return _mmap_reconstruct, (self.data, self.memos)

    def _close(self) -> None:
        for m in self.memos:
            m.close()


def _mmap_reconstruct(data: bytes, memos: list['_Mmap']) -> _ChildProxy:
    buffers = [m.buf for m in memos]
    obj = pickle.loads(data, buffers=buffers)
    return _ChildProxy(obj, *memos)


class _ShmemParentProxy[*Ts, R](_ParentProxy[*Ts, R]):
    __slots__ = ('data', 'memos')

    def __init__(self, call: Callable[[*Ts], R]) -> None:
        super().__init__()
        self.data, buffers = _dumps5(call)
        self.memos: list[tuple[SharedMemory, int]] = []
        for buf in buffers:
            with buf.raw() as m:
                sm = _SharedMemory(create=True, size=m.nbytes)
                weakref.finalize(self, _unlink_shmem, sm)
                if sm.buf is None:
                    raise RuntimeError(f'{sm} is not initialized')
                sm.buf[:] = m
                # Windows: SHM could increase sm.size to page size
                self.memos.append((sm, m.nbytes))

    def __reduce__(self) -> tuple:
        return _shm_reconstruct, (self.data, *self.memos)

    def _close(self) -> None:
        for sm, _ in self.memos:
            sm.close()
            _unlink_shmem(sm)


def _shm_reconstruct(
    data: bytes, *memos: tuple[SharedMemory, int]
) -> _ChildProxy:
    buffers = []
    for sm, size in memos:
        if sm.buf is None:
            raise RuntimeError(f'{sm} is not initialized')
        buffers.append(sm.buf[:size])
    obj = pickle.loads(data, buffers=buffers)
    return _ChildProxy(obj, *memos)


def _unlink_shmem(sm: SharedMemory) -> None:
    try:
        sm.unlink()
    except FileNotFoundError:
        pass


# -------------------------------- reduction ---------------------------------


class _SharedMemory(SharedMemory):
    _fd: int

    def __del__(self) -> None:
        try:
            self.close()
        except BufferError:
            # Exported views retain the mmap and close it when released.
            # SharedMemory.close() stops before closing the POSIX descriptor.
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
        except OSError:
            pass


class _Mmap:
    __slots__ = ('__weakref__', 'buf', 'size', 'tag')

    def __init__(
        self, size: int, tag: str, buf: mmap.mmap | None = None
    ) -> None:
        if buf is None:
            if sys.platform == 'win32':
                buf = mmap.mmap(-1, size, tag, access=mmap.ACCESS_READ)
            else:
                path = (_SHM_ROOT / f'psm_{tag}').as_posix()
                fd = os.open(path, os.O_RDONLY, 0o600)
                try:
                    # resource_tracker.register(path, 'file')
                    buf = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                except:
                    os.close(fd)
                    raise
                weakref.finalize(buf, os.close, fd)

        self.size = size
        self.tag = tag
        self.buf = buf

    def __reduce__(self):
        return type(self), (self.size, self.tag)

    def __sizeof__(self) -> int:
        return self.size

    def close(self) -> None:
        self.buf.close()


def _new_mmap(data: memoryview, tag: str) -> _Mmap:
    tag = f'shm-{os.getpid()}-{tag}'
    size = data.nbytes

    if sys.platform == 'win32':
        buf = mmap.mmap(-1, size, tag, access=mmap.ACCESS_WRITE)
    else:
        path = (_SHM_ROOT / f'psm_{tag}').as_posix()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            # resource_tracker.register(path, 'file')
            os.ftruncate(fd, size)
            buf = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        except:
            os.close(fd)
            os.unlink(path)
            raise

        weakref.finalize(buf, os.unlink, path)
        # weakref.finalize(buf, resource_tracker.maybe_unlink, path, 'file')
        weakref.finalize(buf, os.close, fd)

    buf[:] = data
    return _Mmap(size, tag, buf)


def _dumps5(obj: object) -> tuple[bytes, list[pickle.PickleBuffer]]:
    buffers: list[pickle.PickleBuffer] = []

    def buffer_callback(buf: pickle.PickleBuffer) -> bool:
        with buf.raw() as view:
            if not view.nbytes:
                return True  # Empty buffers  must stay in-band.
        buffers.append(buf)
        return False

    fp = io.BytesIO()
    p = pickle.Pickler(fp, -1, buffer_callback=buffer_callback)
    p.dispatch_table = copyreg.dispatch_table | reducers
    p.dump(obj)
    return fp.getvalue(), buffers


# --------------------------------- settings ---------------------------------


def _get_shm_dir() -> Path:
    if sys.platform != 'win32':
        try:
            stats = os.statvfs(_SYSTEM_SHM)
            if stats.f_bsize * stats.f_bavail > _SYSTEM_SHM_MIN_SIZE:
                return _SYSTEM_SHM
        except OSError:
            pass
    return _SYSTEM_TEMP


def _torch_hook(torch) -> None:
    reducers.update(
        {torch.Tensor: torch.multiprocessing.reductions.reduce_tensor}
        | dict.fromkeys(
            torch.storage._StorageBase.__subclasses__(),
            torch.multiprocessing.reductions.reduce_storage,
        )
    )


_SYSTEM_SHM_MIN_SIZE = int(2e9)  # 2 GB
_SYSTEM_SHM = Path('/dev/shm')
_SYSTEM_TEMP = Path(tempfile.gettempdir())
_SHM_ROOT = _get_shm_dir()

reducers: dict[type, Callable] = {}
loky.set_loky_pickler('pickle')
register_post_import_hook(_torch_hook, 'torch')
