__all__ = [
    'buffered',
    'get_executor',
    'map_n',
    'map_n_dict',
    'max_cpu_count',
    'starmap_n',
]

import atexit
import os
import signal
import sys
import warnings
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import Executor, Future
from contextlib import ExitStack, contextmanager
from cProfile import Profile
from functools import partial
from itertools import batched, chain, islice, repeat, starmap
from logging import getLogger
from multiprocessing import dummy
from multiprocessing.managers import BaseManager
from operator import methodcaller
from pstats import Stats
from queue import SimpleQueue
from threading import Lock
from time import monotonic
from typing import Self, cast

import loky

try:
    import psutil
except ImportError:
    psutil = None

from ._dev import hide_frame
from ._locking import AbsEvent, AbsManager, AbsQueue, f_result, q_get
from ._more import ilen
from ._reduction import move_to_shmem, reducers
from ._thread_quota import ThreadQuota
from ._types import Empty, Some, Unary, empty

_TOTAL_CPUS = os.process_cpu_count()
_NUM_CPUS = _TOTAL_CPUS or 0

if (_env_cpus := os.getenv('GLOW_CPUS')) is not None:
    _NUM_CPUS = min(_NUM_CPUS, int(_env_cpus))
    _NUM_CPUS = max(_NUM_CPUS, 0)

_IDLE_WORKER_TIMEOUT = 10
# TODO: investigate whether this improves load
_FAST_GROW = False
_GRANULAR_SCHEDULING = False
_debug2 = partial(getLogger(__name__).debug, stacklevel=3)

# ------------------- some useful interfaces and functions -------------------


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
def get_executor(max_workers: int, *, mp: bool) -> Generator[Executor]:
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
            threads.shutdown(cancel_futures=True)


def _get_manager(executor: Executor) -> AbsManager:
    return (
        executor._context.Manager()
        if isinstance(executor, loky.ProcessPoolExecutor)
        else dummy
    )


# -------- bufferize iterable by offloading to another thread/process --------


def _consume[T](
    items: Iterable[T], buf: AbsQueue[T | Empty], stop: AbsEvent
) -> None:
    try:
        for item in items:
            if stop.is_set():
                break
            buf.put(item)
    finally:
        buf.put(empty)  # Signal to stop iteration
        buf.put(empty)  # Match last q.get


class buffered[T]:  # noqa: N801
    """Iterate in background thread with at most `latency` items ahead."""

    __slots__ = ('__weakref__', '_consume', '_q', 'close')

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

        ev: AbsEvent = mgr.Event()
        q: AbsQueue[T | Empty] = mgr.Queue(latency)
        self._consume = executor.submit(_consume, iterable, q, ev)
        self._q = q

        # If main is killed, unblocks consumer to allow it to check stop flag
        # Otherwise collects 2nd _empty from q.
        # Called 2nd
        s.callback(q_get, self._q)

        # If main is killed, notifies consumer to stop.
        # If consumer is already stopped (on error or not), does nothing.
        # Called 1st
        s.callback(ev.set)

        self.close = weakref.finalize(self, s.close)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self.close.alive:
            if (item := q_get(self._q)) is not empty:
                return item

            self.close()
            # Reraise exception from source iterable if any
            obj = f_result(self._consume, cancel=False)
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

    def update(self, n: int, start_time: float, fut: Future) -> None:
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
                _debug2(f'Increasing batch size to {size}')

            elif (
                self.duration <= self.MAX_DURATION  # Range is optimal
                or self.size == 1  # Cannot reduce already minimal batch
            ):
                return

            else:  # Too high latency
                size = self._new_scale()
                _debug2(f'Reducing batch size to {size}')

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
) -> Generator[F]:
    for chunk in batched(args_zip, chunksize, strict=False):
        f = submit_chunk(*chunk)
        _debug2(f'Submit {len(chunk)}')
        yield f


def _schedule_auto[F: Future](
    submit_chunk: Callable[..., F],
    args_zip: Iterable[Iterable],
    max_workers: int,
) -> Generator[F]:
    # For the whole wave make futures with the same job size
    size = _AutoSize()
    args_zip_it = iter(args_zip)
    while tuples := [*islice(args_zip_it, size.suggest() * max_workers)]:
        chunksize = len(tuples) // max_workers or 1
        for chunk in batched(tuples, chunksize, strict=False):
            f = submit_chunk(*chunk)
            _debug2(f'Submit {len(chunk)}')
            f.add_done_callback(partial(size.update, len(chunk), monotonic()))
            yield f


def _schedule_auto_v2[F: Future](
    submit_chunk: Callable[..., F], args_zip: Iterable[Iterable]
) -> Generator[F]:
    # Vary job size from future to future
    size = _AutoSize()
    args_zip_it = iter(args_zip)
    while chunk := [*islice(args_zip_it, size.suggest())]:
        f = submit_chunk(*chunk)
        _debug2(f'Submit {len(chunk)}')
        f.add_done_callback(partial(size.update, len(chunk), monotonic()))
        yield f


def _futures_to_results[T](
    s: ExitStack,
    fq: AbsQueue[Future[T]],  # queue to fetch done futures from
    sched_it: Iterable,  # <- `next()` on this could submit future
    on_yield: Unary[T],
) -> Generator[T]:
    with s, hide_frame:  # hide this frame for error in `sched_it.__next__()`
        for _ in sched_it:
            # Retrieve done task
            obj = f_result(q_get(fq))
            if not isinstance(obj, Some):
                with hide_frame:
                    raise obj

            on_yield(obj.x)
            yield obj.x


def _make_task_queue[F: Future](
    unordered: bool,
) -> tuple[Unary[F, None], SimpleQueue[F]]:
    q = SimpleQueue[F]()

    # In `unordered` mode `q` contains only "DONE" tasks,
    # else there are also "PENDING" and "RUNNING" tasks.
    # FIXME: unordered=True -> random freezes (in q.get -> Empty)
    q_put = cast(
        'Unary[F, None]',
        methodcaller('add_done_callback', q.put) if unordered else q.put,
    )
    return q_put, q


def _prefetch[F: Future](
    fs: Iterator[F],
    on_submit: Unary[F],
    n: int | None,
    on_stop: Callable[[], None],
) -> Iterator:
    try:
        sched = map(on_submit, fs)
        qsize = ilen(islice(sched, n))
    except BaseException:
        on_stop()
        raise
    if qsize <= 0:  # Empty `fs`
        on_stop()
        return iter(())
    _debug2(f'Prefetched {qsize} jobs')

    # During iteration skips 1st submit (cause it's already done),
    # then schedules remaining `N-qlen` tasks,
    # and then steps `qlen-1` times to empty extract remaining tasks.
    return chain([None], sched, repeat(None, qsize - 1))


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
    unordered: bool = False,
) -> Iterator[T]:
    """Equivalent to itertools.starmap(fn, iterable).

    Return an iterator whose values are returned from the function evaluated
    with an argument tuple taken from the given sequence.

    Options:
    - workers - Count of workers, by default all hardware threads are occupied.
    - prefetch - Count of extra jobs to schedule over N workers.
      Helps with CPU stalls in ordered mode.
      Increase if job execution time is highly variable.
    - mp - Whether use processes or threads.
    - chunksize - The size of the chunks the iterable will be broken into
      before being passed to a processes.
      Estimated automatically.
      Ignored when threads are used.
    - unordered - Retrieve results in order of completion or in original order.
      In this mode `prefetch` is meaningless, because when some job became done
      it yielded immediately releasing buffer for new job to schedule.
      So no CPU stalls.

    Unlike multiprocessing.Pool or concurrent.futures.Executor this one:
    - never deadlocks on any exception or Ctrl-C interruption.
    - accepts infinite iterables due to lazy task creation.
    - has single interface for both threads and processes.
    - TODO: serializes array-like data using out-of-band Pickle 5 buffers.
    - call immediately creates pool ready to yield results
      (which could take some time cause of serialization for multiprocessing),
      so first `__next__` runs on warmed up pool.
    """
    if max_workers is None:
        max_workers = max_cpu_count(_NUM_CPUS, mp=mp)

    if not max_workers or not _NUM_CPUS:
        return starmap(func, iterable)  # Fallback to single thread

    if mp and chunksize is None and prefetch is None:
        msg = 'With multiprocessing either chunksize or prefetch should be set'
        raise ValueError(msg)

    if unordered:
        prefetch = max(max_workers, 1)
    elif prefetch is not None:
        prefetch = max(prefetch + max_workers, 1)

    s = ExitStack()
    submit = s.enter_context(get_executor(max_workers, mp=mp)).submit

    if mp:
        func = move_to_shmem(func)
    else:
        chunksize = chunksize or 1

    qput, fq = _make_task_queue(unordered)

    if chunksize == 1:
        submit_1 = cast('Callable[..., Future[T]]', partial(submit, func))
        f1s = starmap(submit_1, iterable)
        sched1 = _prefetch(f1s, qput, prefetch, on_stop=s.close)
        return _futures_to_results(s, fq, sched1, lambda _: _debug2('Done 1'))

    submit_n = cast(
        'Callable[..., Future[list[T]]]', partial(submit, _batch_invoke, func)
    )
    if chunksize is not None:
        # Fixed chunksize
        fs = _schedule(submit_n, iterable, chunksize)
    elif not _GRANULAR_SCHEDULING:
        # Dynamic chunksize scaling, submit tasks in waves
        fs = _schedule_auto(submit_n, iterable, max_workers)
    else:
        # Dynamic chunksize scaling
        fs = _schedule_auto_v2(submit_n, iterable)

    sched = _prefetch(fs, qput, prefetch, on_stop=s.close)
    chunks = _futures_to_results(
        s, fq, sched, lambda xs: _debug2(f'Done {len(xs)} items')
    )
    return chain.from_iterable(chunks)


def map_n[T](
    func: Callable[..., T],
    /,
    *iterables: Iterable,
    max_workers: int | None = None,
    prefetch: int | None = 2,
    mp: bool = False,
    chunksize: int | None = None,
    unordered: bool = False,
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
        unordered=unordered,
    )


def map_n_dict[K, T, T2](
    func: Unary[T, T2],
    obj: Mapping[K, T],
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
