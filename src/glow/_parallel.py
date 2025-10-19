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
import logging
import os
import signal
import sys
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from concurrent.futures import Executor, Future
from contextlib import ExitStack, contextmanager
from cProfile import Profile
from functools import partial
from itertools import chain, islice, starmap
from multiprocessing.managers import BaseManager
from operator import methodcaller
from pstats import Stats
from queue import Empty, SimpleQueue
from threading import Lock
from time import monotonic, sleep
from typing import Final, Protocol, Self, cast

import loky

try:
    import psutil
except ImportError:
    psutil = None

from ._dev import hide_frame
from ._more import chunked, ilen
from ._reduction import move_to_shmem, reducers
from ._thread_quota import ThreadQuota
from ._types import Get, Some

_LOGGER = logging.getLogger(__name__)

_TOTAL_CPUS = (
    os.process_cpu_count() if sys.version_info >= (3, 13) else os.cpu_count()
)
_NUM_CPUS = _TOTAL_CPUS or 0

if (_env_cpus := os.getenv('GLOW_CPUS')) is not None:
    _NUM_CPUS = min(_NUM_CPUS, int(_env_cpus))
    _NUM_CPUS = max(_NUM_CPUS, 0)

_IDLE_WORKER_TIMEOUT = 10
# TODO: investigate whether this improves load
_FAST_GROW = False
_GRANULAR_SCHEDULING = False


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


def _torch_limit() -> int | None:
    # Windows platform lacks memory overcommit, so it's sensitive to VMS growth
    if sys.platform != 'win32':
        return None

    torch = sys.modules.get('torch')
    if torch is None or (torch.version.cuda or '') >= '11.7.0':
        # It's expected that torch will fix .nv_fatb readonly flag in its DLLs
        # See https://stackoverflow.com/a/69489193/9868257
        return None

    if psutil is None:
        warnings.warn(
            'Max process count may be calculated incorrectly, '
            'leading to application crash or even BSOD. '
            'Install psutil to avoid that',
            stacklevel=3,
        )
        return None

    # Windows has no overcommit, checking how much processes fit into VMS
    vms: int = psutil.Process().memory_info().vms
    free_vms: int = psutil.virtual_memory().free + psutil.swap_memory().free
    return free_vms // vms


def max_cpu_count(limit: int | None = None, *, mp: bool = False) -> int:
    limits = [_TOTAL_CPUS or 1]

    if limit is not None:
        limits.append(limit)

    if mp and (torch_limit := _torch_limit()) is not None:
        limits.append(torch_limit)

    return min(limits)


_PATIENCE = 0.01


class _TimeoutCallable[T](Protocol):
    def __call__(self, *, timeout: float) -> T: ...


def _retrying[T](f: _TimeoutCallable[T], *exc: type[BaseException]) -> T:
    # See issues
    # https://github.com/dask/dask/pull/2144#issuecomment-290556996
    # https://github.com/dask/dask/pull/2144/files
    # https://github.com/python/cpython/issues/74157
    # FIXED in py3.15+
    while True:
        try:
            return f(timeout=_PATIENCE)
        except exc:
            sleep(0)  # Force switch to another thread to proceed


if sys.platform == 'win32':

    def _exception[T](f: Future[T], /) -> BaseException | None:
        return _retrying(f.exception, TimeoutError)

else:
    _exception = Future.exception


def _result[T](f: Future[T], cancel: bool = True) -> Some[T] | BaseException:
    try:
        return exc if (exc := _exception(f)) else Some(f.result())
    finally:
        if cancel:
            f.cancel()
        del f


def _q_get_fn[T](q: _Queue[T]) -> Get[T]:
    if sys.platform != 'win32':
        return q.get
    return partial(_retrying, q.get, Empty)


# ---------------------------- pool initialization ----------------------------


def _mp_profile() -> None:
    """Multiprocessed profiler."""
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
def get_executor(max_workers: int, *, mp: bool) -> Iterator[Executor]:
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
    """Iterate in background thread with at most `latency` items ahead."""

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
            executor = s.enter_context(get_executor(1, mp=mp))

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

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self.close.alive:
            if (item := self._next()) is not _empty:
                return item

            self.close()
            # Reraise exception from source iterable if any
            obj = _result(self._consume, cancel=False)
            if not isinstance(obj, Some):
                with hide_frame:
                    raise obj

        raise StopIteration


# ---------------------------- automatic batching ----------------------------


class _AutoSize:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0
    size: int = 1
    duration: float = 0.0

    def __init__(self) -> None:
        self.lock = Lock()
        assert self.MIN_DURATION * 2 < self.MAX_DURATION, 'Range is too tight'

    def suggest(self) -> int:
        with self.lock:
            return self.size

    def update(self, n: int, start_time: float, fut: Future[Sized]) -> None:
        # Compute as soon as future became done, discard later if not needed
        duration = monotonic() - start_time

        if fut.cancelled():  # Job never run, zero load
            return

        with self.lock:
            if n != self.size:  # Ran with old size
                return

            # Do EMA smoothing
            self.duration = (
                (0.8 * self.duration + 0.2 * duration)
                if self.duration > 0
                else duration
            )
            if self.duration <= 0:  # Smh not initialized yet
                return  # Or duration is less then `monotonic()` precision

            if self.duration < self.MIN_DURATION:  # Too high IPC overhead
                size = self._new_scale() if _FAST_GROW else self.size * 2
                _LOGGER.debug('Increasing batch size to %d', size)

            elif (
                self.duration <= self.MAX_DURATION  # Range is optimal
                or self.size == 1  # Cannot reduce already minimal batch
            ):
                return

            else:  # Too high latency
                size = self._new_scale()
                _LOGGER.debug('Reducing batch size to %d', size)

            self.size = size
            self.duration = 0.0

    def _new_scale(self) -> int:
        factor = 2 * self.MIN_DURATION / self.duration
        factor = min(factor, 32)
        size = int(self.size * factor)
        return max(size, 1)


# ---------------------- map iterable through function ----------------------


def _schedule[F: Future](
    submit_chunk: Callable[..., F],
    args_zip: Iterable[Iterable],
    chunksize: int,
) -> Iterator[F]:
    for chunk in chunked(args_zip, chunksize):
        f = submit_chunk(*chunk)
        _LOGGER.debug('Submit %d', len(chunk))
        yield f


def _schedule_auto[F: Future](
    submit_chunk: Callable[..., F],
    args_zip: Iterator[Iterable],
    max_workers: int,
) -> Iterator[F]:
    # For the whole wave make futures with the same job size
    size = _AutoSize()
    while tuples := [*islice(args_zip, size.suggest() * max_workers)]:
        chunksize = len(tuples) // max_workers or 1
        for chunk in chunked(tuples, chunksize):
            f = submit_chunk(*chunk)
            _LOGGER.debug('Submit %d', len(chunk))
            f.add_done_callback(partial(size.update, len(chunk), monotonic()))
            yield f


def _schedule_auto_v2[F: Future](
    submit_chunk: Callable[..., F], args_zip: Iterator[Iterable]
) -> Iterator[F]:
    # Vary job size from future to future
    size = _AutoSize()
    while chunk := [*islice(args_zip, size.suggest())]:
        f = submit_chunk(*chunk)
        _LOGGER.debug('Submit %d', len(chunk))
        f.add_done_callback(partial(size.update, len(chunk), monotonic()))
        yield f


def _get_unwrap_iter[T](
    s: ExitStack,
    qsize: int,
    get_f: Get[Future[T]],
    sched_iter: Iterator,
) -> Iterator[T]:
    with s:
        if not qsize:  # No tasks to do
            return

        # Unwrap 1st / schedule `N-qsize` / unwrap `qsize-1`
        with hide_frame:
            for _ in chain([None], sched_iter, range(qsize - 1)):
                # Retrieve done task, exactly `N` calls
                obj = _result(get_f())
                if not isinstance(obj, Some):
                    with hide_frame:
                        raise obj
                yield obj.x


def _enqueue[T](
    fs: Iterable[Future[T]],
    unordered: bool,
) -> tuple[Iterator, Get[Future[T]]]:
    q = SimpleQueue[Future[T]]()

    # In `unordered` mode `q` contains only "DONE" tasks,
    # else there are also "PENDING" and "RUNNING" tasks.
    # FIXME: unordered=True -> random freezes (in q.get -> Empty)
    q_put = cast(
        'Callable[[Future[T]], None]',
        methodcaller('add_done_callback', q.put) if unordered else q.put,
    )
    q_get = _q_get_fn(q)

    # On each `next()` schedules new task
    sched_iter = map(q_put, fs)

    return sched_iter, q_get


def _prefetch(s: ExitStack, sched_iter: Iterator, prefetch: int | None) -> int:
    try:
        # Fetch up to `prefetch` tasks to pre-fill `q`
        qsize = ilen(islice(sched_iter, prefetch))
    except BaseException:
        # Unwind stack here on an error
        s.close()
        raise
    else:
        _LOGGER.debug('Prefetched %d jobs', qsize)
        return qsize


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
    """Equivalent to itertools.starmap(fn, iterable).

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

    TODO: replace `order=True` with `unordered=False`
    """
    if max_workers is None:
        max_workers = max_cpu_count(_NUM_CPUS, mp=mp)

    if not max_workers or not _NUM_CPUS:
        return starmap(func, iterable)  # Fallback to single thread

    if mp and chunksize is None and prefetch is None:
        msg = 'With multiprocessing either chunksize or prefetch should be set'
        raise ValueError(msg)

    if prefetch is not None:
        prefetch = max(prefetch + max_workers, 1)

    unordered = not order

    it = iter(iterable)
    s = ExitStack()
    submit = s.enter_context(get_executor(max_workers, mp=mp)).submit

    if mp:
        func = move_to_shmem(func)
    else:
        chunksize = chunksize or 1

    if chunksize == 1:
        submit_1 = cast('Callable[..., Future[T]]', partial(submit, func))
        f1s = starmap(submit_1, it)
        sched1_iter, get_f = _enqueue(f1s, unordered)
        qsize = _prefetch(s, sched1_iter, prefetch)
        return _get_unwrap_iter(s, qsize, get_f, sched1_iter)

    submit_n = cast(
        'Callable[..., Future[list[T]]]', partial(submit, _batch_invoke, func)
    )
    if chunksize is not None:
        # Fixed chunksize
        fs = _schedule(submit_n, it, chunksize)
    elif not _GRANULAR_SCHEDULING:
        # Dynamic chunksize scaling, submit tasks in waves
        fs = _schedule_auto(submit_n, it, max_workers)
    else:
        # Dynamic chunksize scaling
        fs = _schedule_auto_v2(submit_n, it)

    sched_iter, get_fs = _enqueue(fs, unordered)
    qsize = _prefetch(s, sched_iter, prefetch)
    chunks = _get_unwrap_iter(s, qsize, get_fs, sched_iter)
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
    """Return iterator equivalent to map(func, *iterables).

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
    """Apply `func` to each value in a mapping in parallel way.

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
