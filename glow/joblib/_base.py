from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable
from multiprocessing.pool import Pool, ThreadPool


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


class _BackendBase(ABC):
    supports_sharedmem = False
    parallel = None
    reducer_callback: Callable | None = None

    def __init__(self, level: int):
        self.level = level

    def configure(self, n_jobs, **_):
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


class ImmediateResult:
    def __init__(self, batch, callback=None):
        self.results = batch()
        if callback:
            callback(self)

    def get(self):
        return self.results


class SequentialBackend(_BackendBase):
    supports_sharedmem = True

    def apply_async(self, func, callback=None):
        return ImmediateResult(func, callback)


class ThreadingBackend(_BackendBase):
    supports_sharedmem = True
    _pool_fn: type[Pool] = ThreadPool
    _pool: Pool

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
