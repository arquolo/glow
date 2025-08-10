from __future__ import annotations  # until 3.10

import gc
import multiprocessing as mp
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from multiprocessing.pool import Pool, ThreadPool
from uuid import uuid4

from loky import cpu_count, process_executor

from ._loky import get_memmapping_executor
from ._multiprocessing import MemmappingPool

MAX_NUM_THREADS_VARS = [
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
    'NUMEXPR_NUM_THREADS'
]


def _in_main_thread():
    return threading.current_thread() is threading.main_thread()
    # return isinstance(threading.current_thread(), threading._MainThread)


class ImmediateResult:
    def __init__(self, batch, callback=None):
        self.results = batch()
        if callback:
            callback(self)

    def get(self):
        return self.results


class _BackendBase(ABC):
    supports_sharedmem = False
    parallel = None
    reducer_callback: Callable | None = None

    def __init__(self, level: int):
        self.level = level

    def configure(self, _n_jobs, **_):
        return self, 1

    def get_nested_backend(self) -> tuple[_BackendBase, None]:
        if self.level == 0:
            return ThreadingBackend(1), None
        return SequentialBackend(1 + self.level), None

    @abstractmethod
    def apply_async(self, func, callback=None):
        pass

    def compute_batch_size(self):
        return 1

    def batch_completed(self, batch_size, duration):
        pass

    def terminate(self):
        pass

    def abort_everything(self):
        pass


class SequentialBackend(_BackendBase):
    supports_sharedmem = True

    def apply_async(self, func, callback=None):
        return ImmediateResult(func, callback)


class AutoBatchingMixin:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0

    _size = 1
    _duration = 0.0

    def compute_batch_size(self):
        if 0 < self._duration < self.MIN_DURATION:
            self._size *= 2
            self._duration = 0.0

        elif self._duration > self.MAX_DURATION:
            size = int(2 * self._size * self.MIN_DURATION / self._duration)
            size = max(size, 1)
            if self._size != size:
                self._duration = 0.0
                self._size = size

        return self._size

    def batch_completed(self, batch_size, duration):
        if batch_size == self._size:
            self._duration = ((0.8 * self._duration + 0.2 * duration)
                              if self._duration != 0.0 else duration)

    def terminate(self):
        super().terminate()
        self._size = 1
        self._duration = 0.0


class ThreadingBackend(_BackendBase):
    supports_sharedmem = True
    _pool = None
    _pool_fn: type[Pool] = ThreadPool

    def configure(self, n_jobs, **kwargs):
        if (n_jobs := self._effective_n_jobs(n_jobs)) != 1:
            gc.collect()
            self._pool = self._pool_fn(n_jobs, **kwargs)
            return self, n_jobs
        return SequentialBackend(self.level), 1

    def _effective_n_jobs(self, n_jobs):
        return n_jobs

    def apply_async(self, func, callback=None):
        return self._pool.apply_async(func, callback=callback)

    def terminate(self):
        super().terminate()
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None

    def abort_everything(self):
        self.terminate()


class MultiprocessingBackend(ThreadingBackend, AutoBatchingMixin,
                             _BackendBase):
    supports_sharedmem = False
    _pool_fn = MemmappingPool

    def _effective_n_jobs(self, n_jobs):
        if (mp.current_process().daemon or
                process_executor._CURRENT_DEPTH > 0 or
                not (_in_main_thread() or self.level == 0)):
            return 1
        return n_jobs


class LokyBackend(AutoBatchingMixin, _BackendBase):
    _workers = None
    _id = None

    def configure(self, n_jobs=1, **kwargs):
        if (n_jobs == 1 or mp.current_process().daemon or
                not (_in_main_thread() or self.level == 0)):
            return SequentialBackend(self.level), 1

        self._id = uuid4().hex

        default_n_threads = str(max(cpu_count() // n_jobs, 1))
        os.environ.setdefault('ENABLE_IPC', '1')
        env = {
            var: os.environ.get(var, default_n_threads)
            for var in MAX_NUM_THREADS_VARS
        }
        self._workers, self._temp_manager = \
            get_memmapping_executor(n_jobs, env=env, **kwargs)
        return self, n_jobs

    def reducer_callback(self):
        self._temp_manager.set_context(self._id)

    def apply_async(self, func, callback=None):
        future = self._workers.submit(func)
        future.get = future.result
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def terminate(self):
        super().terminate()
        if self._temp_manager is not None:
            self._temp_manager.unlink(self._id)
        self._temp_manager = self._workers = None

    def abort_everything(self):
        self._workers.shutdown(kill_workers=True)
        with self._workers._submit_resize_lock:
            self._temp_manager.unregister_all()
        self._temp_manager = self._workers = None
