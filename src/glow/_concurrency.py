__all__ = [
    'call_once',
    'shared_call',
    'streaming',
    'threadlocal',
    'weak_memoize',
]

import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, wait
from functools import partial, update_wrapper
from itertools import count
from logging import getLogger
from multiprocessing import Process
from multiprocessing import Queue as MpQueue
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Never, cast, overload
from warnings import warn

from ._cache import memoize
from ._dev import hide_frame
from ._futures import (
    BatchDecorator,
    BatchFn,
    Job,
    PsBatchDecorator,
    UsableSize,
    dispatch,
    gather_fs,
    get_trimmer,
)
from ._locking import q_get
from ._parallel import max_cpu_count
from ._types import Get, Maybe, Some

_LOGGER = getLogger(__name__)


def threadlocal[**P, T](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Get[T]:
    """Create thread-local singleton factory function (functools.partial)."""
    local_ = threading.local()

    def wrapper() -> T:
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

type _JobQueue[T, R] = SimpleQueue[Job[T, R]]


def _build_batches[T, R](
    q: SimpleQueue[Job[T, R]], usable_size: UsableSize[T], latency: float
) -> Iterator[list[Job[T, R]]]:
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
            _LOGGER.debug(f'worker timed out {latency:.3f}s - qd {len(batch)}')
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
    batch_size: int | UsableSize | None = ...,
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
    batch_size: int | UsableSize[T] | None = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchFn[T, R]: ...


def streaming[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] | None = None,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | PsBatchDecorator[T] | BatchFn[T, R]:
    """Delay start of computation to until batch is collected.

    Accepts two timeouts (in seconds):
    - `timeout` is a time to wait till the batch is full, i.e. latency.
    - `pool_timeout` is time to wait for results.

    Also if `batch_size` is not set, or set to 0, only timeout is used.

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
        batch_size = get_trimmer(batch_size)
    q = _start_fetch_compute(func, workers, batch_size, timeout)

    def wrapper(items: Sequence[T]) -> Sequence[R]:
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
        rs, err = gather_fs(enumerate(fs))
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
    batch_size: int,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | BatchFn[T, R]:
    if func is None:
        deco = partial(
            streaming2,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout,
        )
        return cast('BatchDecorator', deco)

    assert callable(func)
    assert workers >= 1

    from ._thread_quota import ThreadQuota  # noqa: PLC0415

    ex = ThreadQuota(workers)
    fut = Future[Sequence[R]]()
    lock = Lock()
    batch: list[T] = []
    deadline = monotonic()

    def _schedule_batch() -> None:
        nonlocal fut
        if batch:
            ex.submit_f(fut, func, batch[:])
            batch.clear()
            fut = Future[Sequence[R]]()

    def sync_late_submit() -> None:
        with lock:
            old_deadline = deadline
            dt = monotonic() - old_deadline

        if dt > 0:
            sleep(dt)

        with lock:
            if deadline is not old_deadline:  # deadline moved forward
                return
            _schedule_batch()

    def sync_submit(x: T) -> tuple[Future[Sequence[R]], int]:
        nonlocal deadline
        with lock:
            if not batch:
                deadline = monotonic() + timeout
                ex.submit(sync_late_submit)

            fut_ = fut
            idx = len(batch)

            batch.append(x)
            if len(batch) == batch_size or monotonic() >= deadline:
                # FIXME: cancel sync_late_submit (how?)
                _schedule_batch()

            return fut_, idx

    def wrapper(xs: Sequence[T]) -> Sequence[R]:
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

        rs, err = gather_fs(zip(fs, fs))
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
        mp: bool = False,
        chunk_size: int = 1,
        latency: float = 0.1,
    ) -> None:
        num_workers = num_workers or max_cpu_count(mp=mp)

        self._ids = count()
        self._worker = _RemoteWorker(func, num_workers + 1)
        self._waiters: dict[int, Lock] = {}
        self._results: dict[int, list[Maybe[R]] | BaseException] = {}

        procs = [
            Process(target=self._worker.run, daemon=True)
            for _ in range(num_workers)
        ]
        for p in procs:
            p.start()

        def shutdown(jobs: MpQueue, procs: Sequence[Process]) -> None:
            for _ in procs:
                jobs.put(None)
            for p in procs:
                p.join()

        self.close = weakref.finalize(self, shutdown, self._worker.jobs, procs)

        Thread(target=self._mover, daemon=True).start()
        self._batch = streaming(
            self._batch_impl, batch_size=chunk_size, timeout=latency
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        [ret] = self._batch([(args, kwargs)])
        if isinstance(ret, Some):
            return ret.x
        raise ret

    def _batch_impl(
        self, akws: Sequence[tuple[tuple, dict]]
    ) -> Sequence[Maybe[R]]:
        if not akws:
            return []

        idx = next(self._ids)
        lk = Lock()
        lk.acquire()
        self._waiters[idx] = lk
        try:
            self._worker.jobs.put((idx, list(akws)))  # IPC, push to worker
        except BaseException:  # failed to serialize
            lk.release()
            self._waiters.pop(idx)
            raise

        with lk:  # wait till owned job is resolved
            rets = self._results.pop(idx)
            if isinstance(rets, BaseException):
                raise rets
            return rets

    def _mover(self) -> Never:
        while True:
            idx, rets = self._worker.results.get()  # IPC, pull from worker
            self._results[idx] = rets
            if lk := self._waiters.pop(idx, None):  # notify waiter
                lk.release()


class _RemoteWorker[**P, R]:
    jobs: MpQueue[tuple[int, list[tuple[tuple, dict]]] | None]
    results: MpQueue[tuple[int, list[Maybe[R]] | BaseException]]

    def __init__(self, func: Callable[P, R], qsize: int) -> None:
        self.func = func
        self.jobs = MpQueue(qsize)
        self.results = MpQueue(qsize)

    def run(self) -> None:
        while True:
            ijobs = self.jobs.get()
            if not ijobs:
                return
            idx, jobs = ijobs
            results: list[Maybe[R]] = []
            for args, kwargs in jobs:
                try:
                    ret = self.func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    results.append(exc)
                else:
                    results.append(Some(ret))

            try:
                self.results.put((idx, results))
            except BaseException as exc:  # noqa: BLE001
                self.results.put((idx, exc))
