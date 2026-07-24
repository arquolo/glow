__all__ = [
    'call_once',
    'shared_call',
    'streaming',
    'threadlocal',
    'weak_memoize',
]

import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, wait
from functools import partial, update_wrapper
from logging import getLogger
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
    get_usable_size,
)
from ._locking import q_get
from ._types import Get

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
) -> BatchFn[T, R]: ...


def streaming[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | PsBatchDecorator[T] | BatchFn[T, R]:
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

    def wrapper(items: Sequence[T]) -> list[R]:
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
