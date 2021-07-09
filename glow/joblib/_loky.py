from __future__ import annotations

__all__ = ['get_memmapping_executor']

import atexit
import multiprocessing as mp
import os
from collections.abc import Callable
from threading import current_thread, main_thread
from uuid import uuid4

from loky import cpu_count
from loky.backend import resource_tracker
from loky.reusable_executor import _ReusablePoolExecutor

from ._base import AutoBatchingMixin, SequentialBackend, _BackendBase
from ._reduction import TYPES, ArrayForwardReducer, reduce_array_backward
from ._resources import _Manager

_IDLE_WORKER_TIMEOUT = 300
_MANAGER = None

MAX_NUM_THREADS_VARS = [
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
    'NUMEXPR_NUM_THREADS'
]


class ExecutorManager(_Manager):
    def drop_all(self):
        while self._cached:
            _, (folder, cleanup) = self._cached.popitem()
            if folder.exists():
                for p in folder.iterdir():
                    resource_tracker.unregister(p.as_posix(), 'file')
            cleanup()
            atexit.unregister(cleanup)


def get_memmapping_executor(n_jobs: int,
                            initializer: Callable | None = None,
                            initargs: tuple = (),
                            env: dict | None = None,
                            max_nbytes: int = 1_000_000,
                            **_) -> tuple[_ReusablePoolExecutor, _Manager]:
    # Create stuff for new Executor
    manager = ExecutorManager()
    forward_reducer = ArrayForwardReducer(
        max_nbytes, unlink_on_gc=True, resolve=manager.resolve)

    executor, executor_is_reused = \
        _ReusablePoolExecutor.get_reusable_executor(
            n_jobs,
            context='loky_init_main',
            timeout=_IDLE_WORKER_TIMEOUT,
            job_reducers=dict.fromkeys(TYPES, forward_reducer),
            result_reducers=dict.fromkeys(TYPES, reduce_array_backward),
            initializer=initializer,
            initargs=initargs,
            env=env)

    global _MANAGER
    if _MANAGER is None or not executor_is_reused:
        # New executor was created, so new reducer was used for it.
        # Set global to it
        _MANAGER = manager
    return executor, _MANAGER


class LokyBackend(AutoBatchingMixin, _BackendBase):
    _workers: _ReusablePoolExecutor
    _manager: _Manager
    _id = None

    def configure(self, n_jobs=1, **kwargs):
        if (n_jobs == 1 or mp.current_process().daemon or
                not (current_thread() is main_thread() or self.level == 0)):
            return SequentialBackend(self.level), 1

        self._id = uuid4().hex

        default_n_threads = str(max(cpu_count() // n_jobs, 1))
        os.environ.setdefault('ENABLE_IPC', '1')
        env = {
            var: os.environ.get(var, default_n_threads)
            for var in MAX_NUM_THREADS_VARS
        }
        self._workers, self._manager = \
            get_memmapping_executor(n_jobs, env=env, **kwargs)
        return self, n_jobs

    def reducer_callback(self):
        self._manager.set_context(self._id)

    def apply_async(self, func, callback=None):
        future = self._workers.submit(func)
        future.get = future.result
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def terminate(self):
        super().terminate()
        if self._manager is not None:
            self._manager.unlink(self._id)
        self._manager = self._workers = None

    def abort_everything(self):
        self._workers.shutdown(kill_workers=True)
        with self._workers._submit_resize_lock:
            self._manager.drop_all()
        self._manager = self._workers = None
