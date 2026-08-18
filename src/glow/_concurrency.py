__all__ = [
    'call_once',
    'shared_call',
    'streaming',
    'threadlocal',
    'weak_memoize',
]

import threading
import weakref
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, wait
from functools import partial, update_wrapper
from itertools import count
from multiprocessing import Process
from multiprocessing import SimpleQueue as MpSimpleQueue
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Never, cast, overload
from warnings import warn

from loguru import logger

from ._cache import memoize
from ._dev import hide_frame
from ._futures import (
    BatchDecorator,
    BatchFn,
    BatchFnRv,
    Job,
    PsBatchDecorator,
    UsableSize,
    dispatch,
    fs_to_results,
    get_usable_size,
)
from ._locking import q_get
from ._parallel import max_cpu_count
from ._types import Get, Maybe, Some


def threadlocal[**P, R](
    fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> Get[R]:
    """Create thread-local singleton factory function (functools.partial)."""
    local_ = threading.local()

    def wrapper() -> R:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)


def call_once[T](fn: Get[T], /) -> Get[T]:
    """Make callable a singleton.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions
    """
    warn(
        'Deprecated. Use `@memoize()` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize()(fn)


def shared_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """Merge duplicate parallel invocations of callable to a single one.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions
    """
    warn(
        'Deprecated. Use `@memoize(0)` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize(0)(fn)


def weak_memoize[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """Preserve each result of each call until they are garbage collected."""
    warn(
        'Deprecated. Use `@memoize(0)` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize(0)(fn)


# ----------------------------- batch collation ------------------------------


def _build_batches[T, R](
    q: SimpleQueue[Job[T, R]], usable_size: UsableSize[T], latency: float
) -> Generator[list[Job[T, R]]]:
    batch = []
    endtime = 0.0

    while True:
        if not batch:
            # Wait indefinitely until the first item is received
            batch = [q_get(q)]
            endtime = monotonic() + latency

        if usable := usable_size([x for x, _ in batch]):
            if usable < len(batch):  # Last append was mistake
                endtime = monotonic() + latency
            yield batch[:usable]
            batch = batch[usable:]
            continue

        try:
            rem = endtime - monotonic()
            batch.append(q.get(timeout=rem) if rem > 0 else q.get(block=False))
        except Empty:
            logger.debug(f'worker timed out {latency:.3f}s - qd {len(batch)}')
            yield batch[:]
            batch = []


def _start_fetch_compute[T, R](
    func: BatchFn[T, R],
    workers: int,
    batch_size: UsableSize[T],
    timeout: float,
) -> SimpleQueue[Job[T, R]]:
    # TODO: Use scalable ThreadPool.
    # Track count of active dispatches and scale workers accordingly
    q = SimpleQueue[Job[T, R]]()
    batching_lock = Lock()
    batches = _build_batches(q, batch_size, timeout)

    def loop() -> Never:
        while True:
            with batching_lock:
                batch = next(batches)
            batch = [x for x in batch if x[1].set_running_or_notify_cancel()]
            if batch:
                dispatch(func, *batch)
            else:
                sleep(0.001)

    for _ in range(workers):
        Thread(target=loop, daemon=True).start()
    return q


@overload
def streaming(
    *,
    batch_size: int | UsableSize = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchDecorator: ...
@overload
def streaming[T](
    *,
    batch_size: UsableSize[T],
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> PsBatchDecorator[T]: ...
@overload
def streaming[T, R](
    func: BatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchFnRv[T, R]: ...


def streaming[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | PsBatchDecorator[T] | BatchFnRv[T, R]:
    """Delay start of computation to until batch is collected.

    Accepts two timeouts (in seconds):
    - `timeout` is a time to wait till the batch is full, i.e. latency.
    - `pool_timeout` is time to wait for results.

    Also if `batch_size` is 0, only timeout is used.

    Uses ideas from
    - https://github.com/ShannonAI/service-streamer
    - https://github.com/leon0707/batch_processor
    - ray.serve.batch
      https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py

    Note: currently supports only functions and bound methods.

    Implementation details:
    - constantly keeps alive N workers
    - any caller enqueues jobs and starts waiting
    - on any failure during waiting caller cancels all jobs it submitted
    - single worker at a time fetches jobs from shared queue, resolves them,
      and notifies all waiters
    """
    if func is None:
        deco = partial(
            streaming,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout,
        )
        return cast('BatchDecorator', deco)

    assert callable(func)
    assert workers >= 1
    if not callable(batch_size):
        batch_size = partial(get_usable_size, batch_size)
    q = _start_fetch_compute(func, workers, batch_size, timeout)

    def wrapper(items: Iterable[T]) -> list[R]:
        fs = {Future[R](): item for item in items}
        try:
            for f, x in fs.items():
                q.put((x, f))  # Schedule task
            dnd = wait(fs, pool_timeout, return_when='FIRST_EXCEPTION')

        finally:  # Cancel all not-yet-running tasks, we're beyond deadline
            for f in fs:
                f.cancel()

        if dnd.not_done:  # Some tasks timed out
            del dnd, fs  # ? Break reference cycle
            raise TimeoutError

        # Cannot time out - all are done
        rs: dict[int, R] = {}
        err = fs_to_results(enumerate(fs), rs)
        if err is None:
            return list(rs.values())
        with hide_frame:
            raise err

    # TODO: if func is instance method - recreate wrapper per instance
    # TODO: find how to distinguish between
    # TODO:  not yet bound method and plain function
    # TODO:  maybe implement __get__ on wrapper
    return update_wrapper(wrapper, func)


def streaming2[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | BatchFnRv[T, R]:
    if func is None:
        deco = partial(
            streaming2,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout,
        )
        return cast(BatchDecorator, deco)

    assert callable(func)
    assert workers >= 1
    assert timeout > 0

    from ._thread_quota import ThreadQuota  # noqa: PLC0415

    ex = ThreadQuota(workers + 1)
    lock = Lock()
    buf: list[T] = []
    futs: list[Future[Sequence[R]]] = [Future()]
    deadlines: list[float] = []
    if not callable(batch_size):
        batch_size = partial(get_usable_size, batch_size)

    def schedule_batch(n: int) -> float | None:
        fut = futs[0]
        batch, buf[:] = buf[:n], buf[n:]
        deadlines.clear()
        if batch:
            ex.submit_f(fut, func, batch)
            futs[0] = Future()
        if buf:
            deadlines[:] = [monotonic() + timeout]
            return timeout
        return None

    def sync_late_submit() -> Never:
        while True:
            with late_lk, lock:
                now = monotonic()
                if deadlines and (sleep_for := deadlines[0] - now) <= 0:
                    sleep_for = schedule_batch(len(buf))
            if sleep_for is None:
                late_lk.acquire()
            else:
                sleep(sleep_for)

    late_lk = Lock()
    late_lk.acquire()
    ex.submit(sync_late_submit)

    def sync_submit(x: T) -> tuple[Future[Sequence[R]], int]:
        with lock:
            now = monotonic()
            if not buf:
                deadlines[:] = [now + timeout]
                late_lk.release()

            fut = futs[0]
            idx = len(buf)
            buf.append(x)

            if n := batch_size(buf):
                schedule_batch(n)
            elif deadlines and now >= deadlines[0]:
                schedule_batch(len(buf))

            return fut, idx

    def wrapper(xs: Iterable[T]) -> list[R]:
        pairs = [sync_submit(x) for x in xs]

        fs = {f for f, _ in pairs}
        try:
            dnd = wait(fs, pool_timeout, return_when='FIRST_EXCEPTION')
        finally:  # Cancel all not-yet-running tasks, we're beyond deadline
            for f in fs:
                f.cancel()
        if dnd.not_done:  # Some tasks timed out
            del dnd, fs  # ? Break reference cycle
            raise TimeoutError

        rs: dict[Future[Sequence[R]], Sequence[R]] = {}
        err = fs_to_results(zip(fs, fs), rs)
        if err is None:
            return [rs[f][i] for f, i in pairs]
        with hide_frame:
            raise err

    return update_wrapper(wrapper, func)


class Remote[**P, R]:
    def __init__(
        self,
        func: Callable[P, R],
        *,
        num_workers: int | None = None,
        chunk_size: int = 1,
        latency: float = 0.1,
        prefetch: int = 1,
    ) -> None:
        num_workers = num_workers or max_cpu_count(mp=True)
        assert chunk_size >= 1
        assert num_workers >= 1
        assert prefetch >= 0
        tmgr = _TaskManager(func, num_workers, prefetch)

        # aggregates calls to batches to pass to each worker
        # `batch_submit` is called from only 1 thread (streaming.workers=1)
        self._batch_submit = streaming(
            tmgr.batch_submit, batch_size=chunk_size, timeout=latency
        )

        # moves results from Mp Queue to per-call mapping
        Thread(target=tmgr.populate_results, daemon=True).start()

        self._results = tmgr.results
        self.close = weakref.finalize(self, tmgr.shutdown)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        [(lk, batch_idx, idx)] = self._batch_submit([(args, kwargs)])

        # Wait till tasks became resolved
        with lk:
            ret = self._results.pop((batch_idx, idx))
        if isinstance(ret, Some):
            return ret.x
        raise ret


type _MpTQueue[*Ts] = MpSimpleQueue[tuple[*Ts] | None]


class _TaskManager[**P, R]:
    def __init__(
        self, func: Callable[P, R], num_workers: int, prefetch: int
    ) -> None:
        self.limit = threading.Semaphore(num_workers + prefetch)
        self.ids = count()
        self.locks: dict[int, Lock] = {}
        self.jobs_mpq: _MpTQueue[int, list[tuple[tuple, dict]]] = (
            MpSimpleQueue()
        )
        self.results_mpq: _MpTQueue[
            int, int, list[Maybe[R]] | BaseException
        ] = MpSimpleQueue()

        # picks jobs from Mp Queue, does compute and puts results to Mp Queue
        self.workers = [
            Process(
                target=_remote_run,
                args=(func, self.jobs_mpq, self.results_mpq),
                daemon=True,
            )
            for _ in range(num_workers)
        ]
        for w in self.workers:
            w.start()

        self.results: dict[tuple[int, int], Maybe[R]] = {}
        self._running = True
        self._run_lock = Lock()

    def batch_submit(
        self, akws: Sequence[tuple[tuple, dict]]
    ) -> list[tuple[Lock, int, int]]:
        # Called from 1 thread, passes new batch of tasks and returns
        if not akws:
            return []
        self.limit.acquire()  # Protect queue from overloading

        with self._run_lock:
            if not self._running:
                self.limit.release()
                raise RuntimeError('cannot submit new task for closed remote')

            batch_idx = next(self.ids)
            lk = Lock()
            lk.acquire()
            self.locks[batch_idx] = lk

            try:
                # Serialize and send (see SimpleQueue impl) to worker [IPC]
                self.jobs_mpq.put((batch_idx, list(akws)))
            except BaseException:  # Serialization failed
                lk.release()
                self.limit.release()
                self.locks.pop(batch_idx)
                raise

        return [(lk, batch_idx, idx) for idx, _ in enumerate(akws)]

    def populate_results(self) -> None:
        stops = 0
        while True:
            out = self.results_mpq.get()  # From worker [IPC]
            if not out:
                stops += 1
                if stops == len(self.workers):
                    break
                continue
            batch_idx, n, rets = out
            self.limit.release()

            if isinstance(rets, BaseException):
                rets = [rets] * n
            for idx, ret in enumerate(rets):
                self.results[batch_idx, idx] = ret

            if lk := self.locks.pop(batch_idx, None):  # Notify waiters
                lk.release()

    def shutdown(self) -> None:  # thread YYY
        with self._run_lock:
            if not self._running:
                return
            self._running = False
            for _ in self.workers:
                self.jobs_mpq.put(None)
        for w in self.workers:
            w.join()


def _remote_run[R](
    func: Callable[..., R],
    jobs_mpq: _MpTQueue[int, list[tuple[tuple, dict]]],
    results_mpq: _MpTQueue[int, int, list[Maybe[R]] | BaseException],
) -> None:
    while ijobs := jobs_mpq.get():  # From main [IPC]
        batch_idx, jobs = ijobs
        rets = [Some.maybe(func, *args, **kwargs) for args, kwargs in jobs]

        n = len(jobs)
        try:
            results_mpq.put((batch_idx, n, rets))  # To main [IPC]
        except BaseException as exc:  # noqa: BLE001
            # Serialization failed
            results_mpq.put((batch_idx, n, exc))  # To main [IPC]

    results_mpq.put(None)  # IPC to main
