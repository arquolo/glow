from __future__ import annotations

__all__ = ['buffered', 'map_n', 'starmap_n']

import atexit
import enum
import os
import signal
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from cProfile import Profile
from dataclasses import dataclass
from functools import partial
from itertools import chain, islice, starmap, tee
from operator import methodcaller
from pstats import Stats
from queue import Queue, SimpleQueue
from threading import Event, RLock
from time import perf_counter
from typing import ContextManager, Protocol, TypeVar, cast

import loky
from loky.process_executor import ProcessPoolExecutor

from ._more import chunked, sliced
from ._reduction import move_to_shmem, reducers

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Future)
_NUM_CPUS = os.cpu_count() or 0
_IDLE_WORKER_TIMEOUT = 10


class _Empty(enum.Enum):
    token = 0


_empty = _Empty.token

# -------------------------- some useful interfaces --------------------------


class _IQueue(Protocol[_T]):
    def get(self) -> _T:
        ...

    def put(self, value: _T) -> None:
        ...


class _IEvent(Protocol):
    def is_set(self) -> bool:
        ...

    def set(self) -> None:  # noqa: A003
        ...


# ---------------------------- pool initialization ----------------------------


def _mp_profile():
    """Multiprocessed profiler"""
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


@contextmanager
def _get_executor(max_workers: int, mp: bool) -> Iterator[Executor]:
    if mp:
        processes: ProcessPoolExecutor = loky.get_reusable_executor(
            max_workers,
            'loky_init_main',
            _IDLE_WORKER_TIMEOUT,
            job_reducers=reducers,
            result_reducers=reducers,
            initializer=_initializer,
        )
        # In generator 'finally' is not reliable enough, use atexit
        hook = atexit.register(processes.shutdown, kill_workers=True)
        yield processes
        atexit.unregister(hook)
    else:
        threads = ThreadPoolExecutor(max_workers)
        try:
            yield threads
        finally:
            # TODO: On pool death, set flag in proxy.
            # TODO: In each worker, check this flag before call,
            # TODO:  if it's set, then kill all worker-owned pools.
            # TODO: To use this, somethere should be dict[ident, list[pool]]
            is_success = sys.exc_info() is None
            threads.shutdown(wait=is_success, cancel_futures=True)


def _setup_queues(
        stack: ExitStack, latency: int,
        executor: bool | Executor) -> tuple[Executor, _IQueue, _IEvent]:
    if not isinstance(executor, Executor):
        executor = stack.enter_context(_get_executor(1, executor))

    if isinstance(executor, ThreadPoolExecutor):
        return executor, Queue(latency), Event()

    assert isinstance(executor, loky.ProcessPoolExecutor)
    mgr = stack.enter_context(executor._context.Manager())
    return executor, mgr.Queue(latency), mgr.Event()


# -------- bufferize iterable by offloading to another thread/process --------


@dataclass
class buffered(Iterable[_T]):  # noqa: N801
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """
    iterable: Iterable[_T]
    latency: int = 2
    mp: bool | Executor = False

    def _consume(self, q: _IQueue[_T | _Empty], stop: _IEvent):
        with ExitStack() as s:
            s.callback(q.put, _empty)  # Match last q.get
            s.callback(q.put, _empty)  # Signal to stop iteration

            for item, _ in zip(self.iterable, iter(stop.is_set, True)):
                q.put(item)

            # if stop.is_set():
            #     s.pop_all()
            # q.put(_empty)  # Will happen if all ok to notify main to stop

    def __iter__(self) -> Iterator[_T]:
        with ExitStack() as s:
            q: _IQueue[_T | _Empty]
            executor, q, stop = _setup_queues(s, self.latency, self.mp)

            s.callback(q.get)  # Wakes q.put when main fails
            s.callback(stop.set)

            task = executor.submit(self._consume, q, stop)
            while (item := q.get()) is not _empty:
                yield item
            task.result()  # Throws if `consume` is dead

    def __len__(self) -> int:
        return len(self.iterable)  # type: ignore


# ---------------------------- automatic batching ----------------------------


class _AutoSize:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0
    size: int = 1
    duration: float = 0.0

    def __init__(self) -> None:
        self.lock = RLock()

    def suggest(self) -> int:
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

    def update(self, start_time: float, fut: Future[list]):
        if fut.cancelled() or fut.exception():
            return

        size = len(fut.result())
        with self.lock:
            if size != self.size:
                return
            duration = perf_counter() - start_time
            self.duration = ((0.8 * self.duration + 0.2 * duration)
                             if self.duration > 0 else duration)


# ---------------------- map iterable through function ----------------------


def _schedule(make_future: Callable[..., _F], args_zip: Iterable[tuple],
              chunksize: int) -> Iterator[_F]:
    return starmap(make_future, chunked(args_zip, chunksize))


def _schedule_auto(make_future: Callable[..., _F], args_zip: Iterable[tuple],
                   max_workers: int) -> Iterator[_F]:
    it = iter(args_zip)
    size = _AutoSize()
    while items := [*islice(it, size.suggest() * max_workers)]:
        chunksize = len(items) // max_workers or 1
        for f in starmap(make_future, sliced(items, chunksize)):
            f.add_done_callback(partial(size.update, perf_counter()))
            yield f


def _unwrap(cm: ContextManager, fs: Iterable[Future[_T]], qsize: int | None,
            order: bool) -> Iterator[_T]:
    q: SimpleQueue[Future] = SimpleQueue()
    q_put = q.put if order else methodcaller('add_done_callback', q.put)

    it1, it2 = tee(fs)
    f_to_none = zip(it1, map(q_put, it2))  # type: ignore
    with cm:
        todo = dict(islice(f_to_none, qsize))
        while todo:
            f = q.get()
            yield f.result()
            todo.pop(f)
            todo.update(islice(f_to_none, 1))


def starmap_n(func: Callable[..., _T],
              iterable: Iterable[tuple],
              /,
              *,
              max_workers: int | None = None,
              prefetch: int | None = None,
              mp: bool = False,
              chunksize: int | None = None,
              order: bool = True) -> Iterator[_T]:
    """
    Equivalent to itertools.starmap(fn, iterable).

    Return an iterator whose values are returned from the function evaluated
    with an argument tuple taken from the given sequence.

    Options:

    - workers - Count of workers, by default all hardware threads are occupied.
    - prefetch - Extra count of scheduled jobs, if not set equals to infinity.
    - mp - Whether use processes or threads.
    - chunksize - The size of the chunks the iterable will be broken into
      before being passed to a processes. Estimated automatically.
      Ignored when threads are used.
    - order - Whether keep results order, or ignore it to increase performance.

    Unlike multiprocessing.Pool or concurrent.futures.Executor this one:

    - never deadlocks on any exception or Ctrl-C interruption.
    - accepts infinite iterables due to lazy task creation (option prefetch).
    - has single interface for both threads and processes.
    - TODO: serializes array-like data using out-of-band Pickle 5 buffers

    Notes:

    - To reduce latency set order to False, order of results will be arbitrary.
    - To increase CPU usage increase prefetch or set it to None.
    - In terms of CPU usage there's no difference between
      prefetch=None and order=False, so choose wisely.
    - Setting order to False makes no use of prefetch more than 0.

    """
    if max_workers is None:
        max_workers = _NUM_CPUS
        if mp and sys.platform == 'win32' and 'torch' in sys.modules:
            # On Windows torch initializes CUDA in each process
            # with RSS leak up to 2GB/process,
            # so we limit count of subprocesses
            max_workers = min(_NUM_CPUS, 12)

    if not max_workers:
        raise ValueError('max_workers should be greater than 0')

    if mp and chunksize is None and prefetch is None:
        raise ValueError('With multiprocessing either chunksize or prefetch '
                         'must be not None')

    if prefetch is not None:
        prefetch += max_workers

    it = iter(iterable)
    cm = ExitStack()
    submit = cm.enter_context(_get_executor(max_workers, mp)).submit

    if mp:
        submit_many = cast(
            Callable[..., Future[list[_T]]],
            partial(submit, move_to_shmem(func)),
        )
        if chunksize is None:
            fs = _schedule_auto(submit_many, it, max_workers)
        else:
            fs = _schedule(submit_many, it, chunksize or 1)

        chunks = _unwrap(cm, fs, prefetch, order)
        return chain.from_iterable(chunks)
    else:
        submit_one = cast(
            Callable[..., Future[_T]],
            partial(submit, func),
        )
        return _unwrap(cm, starmap(submit_one, it), prefetch, order)


def map_n(func: Callable[..., _T],
          /,
          *iterables: Iterable,
          max_workers: int | None = None,
          prefetch: int | None = 2,
          mp: bool = False,
          chunksize: int | None = None,
          order: bool = True) -> Iterator[_T]:
    """
    Returns iterator equivalent to map(func, *iterables).

    Make an iterator that computes the function using arguments from
    each of the iterables. Stops when the shortest iterable is exhausted.

    For extra options, see starmap_n, which is used under hood.
    """
    return starmap_n(
        func,
        zip(*iterables),
        max_workers=max_workers,
        prefetch=prefetch,
        mp=mp,
        chunksize=chunksize,
        order=order)
