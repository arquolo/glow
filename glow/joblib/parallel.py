from __future__ import annotations

__all__ = ['Parallel', 'delayed']

import multiprocessing as mp
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import islice
from threading import RLock
from unittest import mock

from .. import sliced
from ._base import SequentialBackend, ThreadingBackend, _BackendBase
from ._loky import LokyBackend
from ._multiprocessing import MultiprocessingBackend

_backend = threading.local()

_DEFAULT_CTX = os.environ.get('MP_CTX')
BACKENDS = {
    'multiprocessing': MultiprocessingBackend,
    'threading': ThreadingBackend,
    'sequential': SequentialBackend,
    'loky': LokyBackend,
}
delayed = partial(partial, partial)


def get_active_backend():
    if (backend_and_jobs := getattr(_backend, 'backend_and_jobs',
                                    None)) is not None:
        backend, _ = backend_and_jobs
        if backend.supports_sharedmem:
            return backend_and_jobs
        return ThreadingBackend(backend.level), 1
    return LokyBackend(0), 1


@dataclass
class BatchedCalls:
    tasks: Sequence
    backend_and_jobs: tuple[_BackendBase, int]
    reducer_callback: Callable | None = None

    def __call__(self):
        backend, n_jobs = self.backend_and_jobs

        if (backend_and_jobs := getattr(_backend, 'backend_and_jobs',
                                        None)) is not None:
            if backend.level is None:
                backend.level = backend_and_jobs[0].level
        else:
            backend.level = 0

        with mock.patch.object(
                _backend, 'backend_and_jobs', (backend, n_jobs), create=True):
            return [task() for task in self.tasks]

    def __len__(self):
        return len(self.tasks)

    def __reduce__(self):
        if self.reducer_callback is not None:
            self.reducer_callback()
        return BatchedCalls, (self.tasks, self.backend_and_jobs)


class Parallel:
    _original_iterator = None
    _iterating = False
    _aborting = False

    def __init__(self,
                 n_jobs=None,
                 backend=None,
                 pre_dispatch=2,
                 max_nbytes=1e6):
        assert n_jobs != 0

        self._batches = deque()
        self._jobs = deque()
        self._lock = RLock()

        context = mp.get_context(_DEFAULT_CTX)

        active_backend, context_n_jobs = get_active_backend()
        n_jobs = (
            n_jobs if n_jobs is not None else
            (1 if backend is not None else context_n_jobs))
        backend = (
            BACKENDS[backend](active_backend.level)
            if backend is not None else active_backend)

        self._backend, self._n_jobs = backend.configure(
            n_jobs, max_nbytes=max_nbytes, context=context)
        self._pre_dispatch = self._n_jobs * pre_dispatch

    def dispatch_next(self, _, batch_size, timestamp):
        self._backend.batch_completed(batch_size,
                                      time.perf_counter() - timestamp)
        with self._lock:
            if (self._original_iterator is not None and
                    not self.dispatch_one_batch(self._original_iterator)):
                self._iterating = False
                self._original_iterator = None

    def dispatch_one_batch(self, iterator):
        ideal_batch_size = self._backend.compute_batch_size()

        with self._lock:
            try:
                batch = self._batches.popleft()
            except IndexError:
                *jobs, = islice(iterator, ideal_batch_size * self._n_jobs)
                if not jobs:
                    return False

                batch_size = len(jobs) // self._n_jobs

                if (batch_size < ideal_batch_size and
                        iterator is self._original_iterator):
                    batch_size //= 10
                batch_size = max(1, batch_size)

                batch, *rest_batches = (
                    BatchedCalls(s, self._backend.get_nested_backend(),
                                 self._backend.reducer_callback)
                    for s in sliced(jobs, batch_size))

                self._batches.extend(rest_batches)

            if len(batch) == 0:
                return False

            if not self._aborting:
                callback = partial(
                    self.dispatch_next,
                    batch_size=len(batch),
                    timestamp=time.perf_counter())
                self._jobs.append(self._backend.apply_async(batch, callback))
            return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._backend is not None:
            self._backend.terminate()
        self._jobs = deque()

    def __call__(self, iterable):
        iterator = iter(iterable)

        if self._n_jobs != 1:
            self._original_iterator = iterator
            iterator = islice(iterator, self._pre_dispatch)

        with self:
            if self.dispatch_one_batch(iterator):
                self._iterating = self._original_iterator is not None
            while self.dispatch_one_batch(iterator):
                pass

            if self._n_jobs == 1:
                self._iterating = False

            while self._iterating or self._jobs:
                if not self._jobs:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    job = self._jobs.popleft()
                try:
                    yield from job.get()
                except BaseException:  # noqa: B902
                    self._aborting = True
                    if self._backend is not None:
                        self._backend.abort_everything()
                    raise
