__all__ = ['mapped']

import atexit
import os
import signal
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import ExitStack
from cProfile import Profile
from dataclasses import dataclass
from functools import partial
from itertools import chain, islice
from pstats import Stats
from threading import RLock
from time import perf_counter
from typing import (Callable, Deque, Iterable, Iterator, Optional, Sequence,
                    Tuple, TypeVar, cast)

import loky

from ._reduction import reducers, serialize
from .more import chunked, sliced

_T = TypeVar('_T')
_NUM_CPUS = os.cpu_count() or 0
_IDLE_WORKER_TIMEOUT = 10

loky.backend.context.set_start_method('loky_init_main')


def _mp_profile():
    prof = Profile()
    prof.enable()

    def _finalize(lines=50):
        prof.disable()
        with open(f'prof-{os.getpid()}.txt', 'w') as fp:
            Stats(prof, stream=fp).sort_stats('cumulative').print_stats(lines)

    atexit.register(_finalize)


def _initializer():
    # `signal.signal` suppresses KeyboardInterrupt in child processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get('_GLOW_MP_PROFILE'):
        _mp_profile()


def _get_pool(num_workers: int) -> Executor:
    return loky.get_reusable_executor(  # type: ignore
        num_workers,
        timeout=_IDLE_WORKER_TIMEOUT,
        job_reducers=reducers,
        result_reducers=reducers,
        initializer=_initializer,
    )


class _AutoSize:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0
    size: int = 1
    duration: float = 0.0

    def __init__(self) -> None:
        self.lock = RLock()

    def suggest(self):
        with self.lock:
            if 0 < self.duration < self.MIN_DURATION:
                self.size *= 2
                self.duration = 0.0

            elif self.duration > self.MAX_DURATION:
                size = int(2 * self.size * self.MIN_DURATION / self.duration)
                size = max(size, 1)
                if self.size != size:
                    self.duration = 0.0
                    self.size = size

            return self.size

    def update(self, size, init, _):
        with self.lock:
            if size != self.size:
                return
            duration = perf_counter() - init
            self.duration = ((0.8 * self.duration + 0.2 * duration)
                             if self.duration > 0 else duration)


@dataclass
class _Mapped(Iterable[_T]):
    proxy: Callable[..., Sequence[_T]]
    iterables: Tuple[Iterable, ...]
    num_workers: int
    latency: int
    mp: bool
    chunksize: Optional[int]

    def _submit(self, executor: Executor,
                iterable: Iterable[tuple],
                chunksize: int) -> Iterator[Future]:
        for chunk in chunked(iterable, chunksize):
            yield executor.submit(self.proxy, *chunk)

    def _submit_auto(self, executor: Executor,
                     iterable: Iterable[tuple]) -> Iterator[Future]:
        iter_ = iter(iterable)
        njobs = self.num_workers
        asize = _AutoSize()

        while items := [*islice(iter_, asize.suggest() * njobs)]:
            for chunk in sliced(items, max(1, len(items) // njobs)):
                if not chunk:
                    continue
                fut = executor.submit(self.proxy, *chunk)
                fut.add_done_callback(
                    partial(asize.update, len(chunk), perf_counter()))
                yield fut

    def _iter_chunks(self, stack: ExitStack,
                     jobs: Iterator[Future]) -> Iterator[Sequence[_T]]:
        results = Deque[Future]()
        stack.callback(lambda: {fut.cancel() for fut in reversed(results)})

        results.extend(islice(jobs, self.latency + self.num_workers))
        while results:
            yield results.popleft().result()
            results.extend(islice(jobs, 1))

    def __iter__(self) -> Iterator[_T]:
        terminator = None
        if self.mp:
            executor = _get_pool(self.num_workers)
            terminator = atexit.register(executor.shutdown, kill_workers=True)
        else:
            executor = ThreadPoolExecutor(self.num_workers)

        with ExitStack() as stack:
            if not self.mp:
                stack.enter_context(executor)

            iterable = zip(*self.iterables)
            if self.mp and self.chunksize is None:
                chunks = self._submit_auto(executor, iterable)
            else:
                chunks = self._submit(executor, iterable, self.chunksize or 1)

            yield from chain.from_iterable(self._iter_chunks(stack, chunks))

        if terminator is not None:
            atexit.unregister(terminator)

    def __len__(self) -> int:
        return min(map(len, self.iterables))  # type: ignore


def mapped(func: Callable[..., _T],
           *iterables: Iterable,
           num_workers: int = _NUM_CPUS,
           latency: int = 2,
           mp: bool = False,
           chunksize: Optional[int] = None) -> _Mapped[_T]:
    """Returns an iterator equivalent to map(fn, *iterables).

    Differences:
    - Uses multiple threads or processes, whether chunksize is zero or not.
    - Unlike multiprocessing.Pool or concurrent.futures.Executor
      *almost* never deadlocks on any exception or Ctrl-C interruption.

    Parameters:
    - fn - A callable that will take as many arguments as there are passed
      iterables.
    - workers - Count of workers, by default all hardware threads are occupied.
    - latency - Count of extra tasks each worker can grab.
    - mp - Whether use multiple processes or threads.
    - chunksize - The size of the chunks the iterable will be broken into
      before being passed to a worker. By default is estimated automatically.

    Calls may be evaluated out-of-order.
    """
    if not num_workers:
        raise ValueError('num_workers should be greater than 0')
        # return cast(_Mapped[_T], map(func, *iterables))

    if not mp and chunksize is not None:
        raise ValueError('In threaded mode chunksize is not supported')

    return _Mapped(
        cast(Callable[..., Sequence[_T]], serialize(func, mp)),
        iterables,
        num_workers=num_workers,
        latency=latency,
        mp=mp,
        chunksize=chunksize,
    )
