__all__ = ['MemmappingPool']

import copyreg
import multiprocessing as mp
import sys
from io import BytesIO
from multiprocessing.context import assert_spawning
from multiprocessing.pool import Pool
from pickle import Pickler
from time import sleep

import numpy as np

from ._reduction import ArrayForwardReducer, reduce_array_backward
from ._resources import TemporaryResourcesManager


class PicklingQueue:
    def __init__(self, context, reducers):
        self._reducers = reducers
        self._reader, self._writer = context.Pipe(duplex=False)
        self._rlock = context.Lock()
        self._wlock = None if sys.platform == 'win32' else context.Lock()
        self._make_methods()

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock,
                self._reducers)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock,
         self._reducers) = state
        self._make_methods()

    def empty(self):
        return not self._reader.poll()

    def _make_methods(self):
        self._recv = recv = self._reader.recv
        racquire, rrelease = self._rlock.acquire, self._rlock.release

        def get():
            racquire()
            try:
                return recv()
            finally:
                rrelease()

        self.get = get

        if self._reducers:

            def send(obj):
                buffer = BytesIO()
                p = Pickler(buffer, -1)
                p.dispatch_table = {**copyreg.dispatch_table, **self._reducers}
                p.dump(obj)
                self._writer.send_bytes(buffer.getvalue())

            self._send = send
        else:
            self._send = send = self._writer.send

        if self._wlock is None:
            self.put = send
        else:
            wlock_acquire, wlock_release = \
                self._wlock.acquire, self._wlock.release

            def put(obj):
                wlock_acquire()
                try:
                    return send(obj)
                finally:
                    wlock_release()

            self.put = put


class MemmappingPool(Pool):
    def __init__(self, processes=None, max_nbytes=1e6, **kwargs):
        self._temp_manager = manager = TemporaryResourcesManager()
        self._temp_manager.init()

        reduce_array_forward = ArrayForwardReducer(max_nbytes, False,
                                                   manager.resolve)
        self._forward_reducers, self._backward_reducers = (
            dict.fromkeys([np.ndarray, np.memmap], fn)
            for fn in (reduce_array_forward, reduce_array_backward))

        super().__init__(processes=processes, **kwargs)

    def _setup_queues(self):
        context = getattr(self, '_ctx', mp)
        self._inqueue = PicklingQueue(context, self._forward_reducers)
        self._outqueue = PicklingQueue(context, self._backward_reducers)
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
        self._temp_manager.unlink_all()
