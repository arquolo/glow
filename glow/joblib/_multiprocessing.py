__all__ = ['MemmappingPool']

import copyreg
import multiprocessing as mp
import sys
from contextlib import nullcontext
from io import BytesIO
from multiprocessing.context import assert_spawning
from multiprocessing.pool import Pool
from pickle import Pickler
from threading import current_thread, main_thread
from time import sleep
from uuid import uuid4

from loky import process_executor

from ._base import AutoBatchingMixin, ThreadingBackend, _BackendBase
from ._reduction import TYPES, ArrayForwardReducer, reduce_array_backward
from ._resources import _Manager


class PicklingQueue:
    def __init__(self, context, reducer):
        self._reducer = reducer
        self._reader, self._writer = context.Pipe(duplex=False)
        self._rlock = context.Lock()
        self._wlock = None if sys.platform == 'win32' else context.Lock()

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock,
                self._reducer)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock,
         self._reducer) = state

    def empty(self):
        return not self._reader.poll()

    def _send(self, obj):
        buffer = BytesIO()
        p = Pickler(buffer, -1)
        p.dispatch_table = {
            **copyreg.dispatch_table,
            **dict.fromkeys(TYPES, self._reducer),
        }
        p.dump(obj)
        self._writer.send_bytes(buffer.getvalue())

    def _recv(self):
        return self._reader.recv()

    def put(self, obj):
        with self._wlock or nullcontext():
            self._send(obj)

    def get(self):
        with self._rlock:
            return self._recv()


class PoolManager(_Manager):
    def __init__(self):
        super().__init__()
        self.set_context(uuid4().hex)

    def drop_all(self):
        for context_id in [*self._cached]:
            self.unlink(context_id)


class MemmappingPool(Pool):
    def __init__(self, processes=None, max_nbytes=1e6, **kwargs):
        self._manager = PoolManager()
        self._fwd_reducer = ArrayForwardReducer(
            max_nbytes, unlink_on_gc=False, resolve=self._manager.resolve)
        super().__init__(processes=processes, **kwargs)

    def _setup_queues(self):
        context = getattr(self, '_ctx', mp)
        self._inqueue = PicklingQueue(context, self._fwd_reducer)
        self._outqueue = PicklingQueue(context, reduce_array_backward)
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv

    def terminate(self):
        for _ in range(10):
            try:
                super().terminate()
                break
            except OSError as e:
                if sys.platform == 'win32' and isinstance(e, WindowsError):
                    sleep(0.1)
        self._manager.drop_all()


class MultiprocessingBackend(ThreadingBackend, AutoBatchingMixin,
                             _BackendBase):
    supports_sharedmem = False
    _pool_fn = MemmappingPool

    def _effective_n_jobs(self, n_jobs):
        if (mp.current_process().daemon or
                process_executor._CURRENT_DEPTH > 0 or
                not (current_thread() is main_thread() or self.level == 0)):
            return 1
        return n_jobs
