__all__ = [
    'buffered',
    'get_executor',
    'map_n',
    'map_n_dict',
    'max_cpu_count',
    'starmap_n',
]

import atexit
import enum
import os
import signal
import sys
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from concurrent.futures import Executor, Future
from concurrent.futures import TimeoutError as _TimeoutError
from contextlib import ExitStack, contextmanager
from cProfile import Profile
from functools import partial
from itertools import chain, islice, starmap
from logging import getLogger
from multiprocessing.managers import BaseManager
from operator import methodcaller
from pstats import Stats
from queue import Empty, SimpleQueue
from threading import Lock
from time import perf_counter, sleep
from typing import Final, Protocol, cast

import loky

try:
    import psutil
except ImportError:
    psutil = None

from ._more import chunked, ilen
from ._reduction import move_to_shmem, reducers
from ._thread_quota import ThreadQuota

_TOTAL_CPUS = (
    os.process_cpu_count() if sys.version_info >= (3, 13) else os.cpu_count()
)
_NUM_CPUS = _TOTAL_CPUS or 0

if (_env_cpus := os.getenv('GLOW_CPUS')) is not None:
    _NUM_CPUS = min(_NUM_CPUS, int(_env_cpus))
    _NUM_CPUS = max(_NUM_CPUS, 0)
_IDLE_WORKER_TIMEOUT = 10
_GRANULAR_SCHEDULING = False  # TODO: investigate whether this improves load
_LOGGER = getLogger(__name__)


class _Empty(enum.Enum):
    token = 0


_empty: Final = _Empty.token

# ------------------- some useful interfaces and functions -------------------


class _Queue[T](Protocol):
    def get(self, block: bool = ..., timeout: float | None = ...) -> T: ...
    def put(self, item: T) -> None: ...


class _Event(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class _Manager(Protocol):
    def Event(self) -> _Event: ...  # noqa: N802
    def Queue(self, /, maxsize: int) -> _Queue: ...  # noqa: N802


def _get_cpu_count_limits(
    upper_bound: int = sys.maxsize, mp: bool = False
) -> Iterator[int]:
    yield upper_bound
    yield _TOTAL_CPUS or 1

    # Windows platform lacks memory overcommit, so it's sensitive to VMS growth
    if not mp or sys.platform != 'win32' or 'torch' not in sys.modules:
        return

    if (sys.modules['torch'].version.cuda or '') >= '11.7.0':
        # It's expected that torch will fix .nv_fatb readonly flag in its DLLs
        # See https://stackoverflow.com/a/69489193/9868257
        return

    if psutil is None:
        warnings.warn(
            'Max process count may be calculated incorrectly, '
            'leading to application crash or even BSOD. '
            'Install psutil to avoid that',
            stacklevel=3,
        )
        return

    # Overcommit on Windows is forbidden, thus VMS planning is necessary
    vms: int = psutil.Process().memory_info().vms
    free_vms: int = psutil.virtual_memory().free + psutil.swap_memory().free
    yield free_vms // vms


def max_cpu_count(upper_bound: int = sys.maxsize, mp: bool = False) -> int:
    return min(_get_cpu_count_limits(upper_bound, mp))


_PATIENCE = 0.01


def _retry_call[T](fn: Callable[..., T], *exc: type[BaseException]) -> T:
    # See issues
    # https://bugs.python.org/issue29971
    # https://github.com/python/cpython/issues/74157
    # https://github.com/dask/dask/pull/2144#issuecomment-290556996
    # https://github.com/dask/dask/pull/2144/files
    while True:
        try:
            return fn(timeout=_PATIENCE)
        except exc:
            sleep(0)  # Force switch to another thread to proceed


if sys.platform == 'win32':

    def _result[T](f: Future[T], /) -> T:
        return _retry_call(f.result, _TimeoutError)

else:
    _result = Future.result


def _result_or_cancel[T](f: Future[T]) -> T:
    try:
        try:
            return _result(f)
        finally:
            f.cancel()
    finally:
        del f


def _q_get_fn[T](q: _Queue[T]) -> Callable[[], T]:
    if sys.platform != 'win32':
        return q.get
    return partial(_retry_call, q.get, Empty)


# ---------------------------- pool initialization ----------------------------


def _mp_profile() -> None:
    """Multiprocessed profiler"""
    prof = Profile()
    prof.enable()

    def _finalize(lines=50) -> None:
        prof.disable()
        with open(f'prof-{os.getpid()}.txt', 'w') as fp:
            Stats(prof, stream=fp).sort_stats('cumulative').print_stats(lines)

    atexit.register(_finalize)


def _initializer() -> None:
    # `signal.signal` suppresses KeyboardInterrupt in child processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get('_GLOW_MP_PROFILE'):
        _mp_profile()


@contextmanager
def get_executor(max_workers: int, mp: bool) -> Iterator[Executor]:
    if mp:
        processes: loky.ProcessPoolExecutor = loky.get_reusable_executor(
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
        threads = ThreadQuota(max_workers)
        try:
            yield threads
        finally:
            is_success = sys.exception() is None
            threads.shutdown(wait=is_success, cancel_futures=True)


def _get_manager(executor: Executor) -> _Manager:
    if isinstance(executor, loky.ProcessPoolExecutor):
        return executor._context.Manager()

    else:  # noqa: RET505
        from multiprocessing.dummy import Manager

        return Manager()


# -------- bufferize iterable by offloading to another thread/process --------


def _consume[T](
    items: Iterable[T], buf: _Queue[T | _Empty], stop: _Event
) -> None:
    try:
        for item in items:
            if stop.is_set():
                break
            buf.put(item)
    finally:
        buf.put(_empty)  # Signal to stop iteration
        buf.put(_empty)  # Match last q.get


class buffered[T](Iterator[T]):  # noqa: N801
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """

    __slots__ = ('__weakref__', '_consume', '_next', 'close')

    def __init__(
        self,
        iterable: Iterable[T],
        /,
        *,
        latency: int = 2,
        mp: bool | Executor = False,
    ) -> None:
        s = ExitStack()
        if isinstance(mp, Executor):
            executor = mp
        else:
            executor = s.enter_context(get_executor(1, mp))

        mgr = _get_manager(executor)
        if isinstance(mgr, BaseManager):
            s.enter_context(mgr)

        ev: _Event = mgr.Event()
        q: _Queue[T | _Empty] = mgr.Queue(latency)
        self._consume = executor.submit(_consume, iterable, q, ev)
        self._next = _q_get_fn(q)

        # If main is killed, unblocks consumer to allow it to check stop flag
        # Otherwise collects 2nd _empty from q.
        # Called 2nd
        s.callback(self._next)

        # If main is killed, notifies consumer to stop.
        # If consumer is already stopped (on error or not), does nothing.
        # Called 1st
        s.callback(ev.set)

        self.close = weakref.finalize(self, s.close)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.close.alive:
            if (item := self._next()) is not _empty:
                return item

            self.close()
            # Reraise exception from source iterable if any
            _result(self._consume)

        raise StopIteration


# ---------------------------- automatic batching ----------------------------


class _AutoSize:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0
    size: int = 1
    duration: float = 0.0

    def __init__(self) -> None:
        self.lock = Lock()

    def suggest(self) -> int:
        with self.lock:
            if 0 < self.duration < self.MIN_DURATION:
                self.size *= 2
                self.duration = 0.0
                _LOGGER.debug('Doubling batch size to %d', self.size)

            elif self.duration > self.MAX_DURATION:
                size = int(2 * self.size * self.MIN_DURATION / self.duration)
                size = max(size, 1)
                if self.size != size:
                    self.duration = 0.0
                    self.size = size
                    _LOGGER.debug('Reducing batch size to %d', self.size)

            return self.size

    def update(self, start_time: float, fut: Future[Sized]) -> None:
        # Compute as soon as future became done, discard later if not needed
        duration = perf_counter() - start_time

        try:
            # Cannot time out, as future is always completed at this point.
            # Though it's unknown whether it succeeded or not, so use EAFP
            r = fut.result()
        except BaseException:  # noqa: BLE001
            return

        with self.lock:
            if len(r) != self.size:
                return
            self.duration = (
                (0.8 * self.duration + 0.2 * duration)
                if self.duration > 0
                else duration
            )


# ---------------------- map iterable through function ----------------------


def _schedule[F: Future](
    make_future: Callable[..., F], args_zip: Iterable[Iterable], chunksize: int
) -> Iterator[F]:
    return starmap(make_future, chunked(args_zip, chunksize))


def _schedule_auto[F: Future](
    make_future: Callable[..., F],
    args_zip: Iterator[Iterable],
    max_workers: int,
) -> Iterator[F]:
    # For the whole wave make futures with the same job size
    size = _AutoSize()
    while tuples := [*islice(args_zip, size.suggest() * max_workers)]:
        chunksize = len(tuples) // max_workers or 1
        for f in starmap(make_future, chunked(tuples, chunksize)):
            f.add_done_callback(partial(size.update, perf_counter()))
            yield f


def _schedule_auto_v2[F: Future](
    make_future: Callable[..., F], args_zip: Iterator[Iterable]
) -> Iterator[F]:
    # Vary job size from future to future
    size = _AutoSize()
    while args := [*islice(args_zip, size.suggest())]:
        f = make_future(*args)
        f.add_done_callback(partial(size.update, perf_counter()))
        yield f


def _get_unwrap_iter[T](
    s: ExitStack,
    qsize: int,
    get_done_f: Callable[[], Future[T]],
    fs_scheduler: Iterator,
) -> Iterator[T]:
    with s:
        if not qsize:  # No tasks to do
            return

        # Unwrap 1st / schedule `N-qsize` / unwrap `qsize-1`
        for _ in chain([None], fs_scheduler, range(qsize - 1)):

            # Retrieve done task, exactly `N` calls
            yield _result_or_cancel(get_done_f())


def _unwrap[T](
    s: ExitStack, fs: Iterable[Future[T]], qsize: int | None, order: bool
) -> Iterator[T]:
    q = SimpleQueue[Future[T]]()

    # If `order`, then `q` has "PENDING"/"RUNNING"/"DONE" tasks,
    # otherwise it only has "DONE" tasks.
    # FIXME: order=False -> random freezes (in q.get -> Empty)
    q_put = cast(
        Callable[[Future[T]], None],
        q.put if order else methodcaller('add_done_callback', q.put),
    )

    # On each `next()` schedules new task
    fs_scheduler = map(q_put, fs)
    try:
        # Fetch up to `qsize` tasks to pre-fill `q`
        qsize = ilen(islice(fs_scheduler, qsize))

    except BaseException:
        # Unwind stack here on an error
        s.close()
        raise

    else:
        return _get_unwrap_iter(s, qsize, _q_get_fn(q), fs_scheduler)


def _batch_invoke[*Ts, R](
    func: Callable[[*Ts], R], *items: tuple[*Ts]
) -> list[R]:
    return [*starmap(func, items)]


def starmap_n[T](
    func: Callable[..., T],
    iterable: Iterable[Iterable],
    /,
    *,
    max_workers: int | None = None,
    prefetch: int | None = 2,
    mp: bool = False,
    chunksize: int | None = None,
    order: bool = True,
) -> Iterator[T]:
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
    - TODO: serializes array-like data using out-of-band Pickle 5 buffers.
    - before first `__next__` call it submits at most `prefetch` jobs
      to warmup pool of workers.

    Notes:

    - To reduce latency set order to False, order of results will be arbitrary.
    - To increase CPU usage increase prefetch or set it to None.
    - In terms of CPU usage there's no difference between
      prefetch=None and order=False, so choose wisely.
    - Setting order to False makes no use of prefetch more than 0.

    TODO: replace `order=True` with `heap=False`
    """
    if max_workers is None:
        max_workers = max_cpu_count(_NUM_CPUS, mp)

    if not max_workers or not _NUM_CPUS:
        return starmap(func, iterable)  # Fallback to single thread

    if mp and chunksize is None and prefetch is None:
        raise ValueError(
            'With multiprocessing either chunksize or prefetch should be set'
        )

    if prefetch is not None:
        prefetch = max(prefetch + max_workers, 1)

    it = iter(iterable)
    s = ExitStack()
    submit = s.enter_context(get_executor(max_workers, mp)).submit

    if mp:
        func = move_to_shmem(func)
    else:
        chunksize = chunksize or 1

    if chunksize == 1:
        submit_one = cast(Callable[..., Future[T]], partial(submit, func))
        return _unwrap(s, starmap(submit_one, it), prefetch, order)

    submit_many = cast(
        Callable[..., Future[list[T]]], partial(submit, _batch_invoke, func)
    )
    if chunksize is not None:
        # Fixed chunksize
        fs = _schedule(submit_many, it, chunksize)
    elif not _GRANULAR_SCHEDULING:
        # Dynamic chunksize scaling, submit tasks in waves
        fs = _schedule_auto(submit_many, it, max_workers)
    else:
        # Dynamic chunksize scaling
        fs = _schedule_auto_v2(submit_many, it)

    chunks = _unwrap(s, fs, prefetch, order)
    return chain.from_iterable(chunks)


def map_n[T](
    func: Callable[..., T],
    /,
    *iterables: Iterable,
    max_workers: int | None = None,
    prefetch: int | None = 2,
    mp: bool = False,
    chunksize: int | None = None,
    order: bool = True,
) -> Iterator[T]:
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
        order=order,
    )


def map_n_dict[K, T1, T2](
    func: Callable[[T1], T2],
    obj: Mapping[K, T1],
    /,
    *,
    max_workers: int | None = None,
    prefetch: int | None = 2,
    mp: bool = False,
    chunksize: int | None = None,
) -> dict[K, T2]:
    """
    Apply `func` to each value in a mapping in parallel way.
    For extra options, see starmap_n, which is used under hood.
    """
    iter_values = map_n(
        func,
        obj.values(),
        max_workers=max_workers,
        prefetch=prefetch,
        mp=mp,
        chunksize=chunksize,
    )
    return dict(zip(obj.keys(), iter_values, strict=True))
