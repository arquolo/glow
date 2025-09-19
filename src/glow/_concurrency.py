__all__ = [
    'call_once',
    'shared_call',
    'streaming',
    'threadlocal',
    'weak_memoize',
]

import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import CancelledError, Future, wait
from functools import partial, update_wrapper
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Never, cast
from warnings import warn

from ._cache import memoize
from ._dev import hide_frame
from ._types import AnyFuture, BatchDecorator, BatchFn, Some

_PATIENCE = 0.01


def threadlocal[T](
    fn: Callable[..., T], /, *args: object, **kwargs: object
) -> Callable[[], T]:
    """Create thread-local singleton factory function (functools.partial)."""
    local_ = threading.local()

    def wrapper() -> T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)


def call_once[T](fn: Callable[[], T], /) -> Callable[[], T]:
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

type _Job[T, R] = tuple[T, Future[R]]


def _fetch_batch[T](
    q: SimpleQueue[T], batch_size: int | None, timeout: float
) -> list[T]:
    batch: list[T] = []

    # Wait indefinitely until the first item is received
    if sys.platform == 'win32':
        # On Windows lock.acquire called without a timeout is not interruptible
        # See:
        # https://bugs.python.org/issue29971
        # https://github.com/dask/dask/pull/2144#issuecomment-290556996
        # https://github.com/dask/dask/pull/2144/files
        while not batch:
            try:
                batch.append(q.get(timeout=_PATIENCE))
            except Empty:
                sleep(0)  # Allow other thread to fill the batch
    else:
        batch.append(q.get())

    endtime = monotonic() + timeout
    remaining = timeout
    while remaining > 0 and (batch_size is None or len(batch) < batch_size):
        try:
            batch.append(q.get(timeout=remaining))
        except Empty:
            break
        remaining = endtime - monotonic()
    return batch


def _batch_invoke[T, R](
    func: BatchFn[T, R], batch: Sequence[_Job[T, R]]
) -> None:
    batch = [(x, f) for x, f in batch if f.set_running_or_notify_cancel()]
    if not batch:
        return

    obj: Some[Sequence[R]] | BaseException
    try:
        with hide_frame:
            obj = Some(func([x for x, _ in batch]))
            if not isinstance(obj.x, Sequence):
                obj = TypeError(
                    f'Call returned non-sequence. Got {type(obj.x).__name__}'
                )
            elif len(obj.x) != len(batch):
                obj = RuntimeError(
                    f'Call with {len(batch)} arguments '
                    f'incorrectly returned {len(obj.x)} results'
                )
    except BaseException as exc:  # noqa: BLE001
        obj = exc

    if isinstance(obj, Some):
        for (_, f), r in zip(batch, obj.x):
            f.set_result(r)
    else:
        for _, f in batch:
            f.set_exception(obj)


def _start_fetch_compute[T, R](
    func: BatchFn[T, R],
    workers: int,
    batch_size: int | None,
    timeout: float,
) -> SimpleQueue[_Job[T, R]]:
    q = SimpleQueue()  # type: ignore[var-annotated]
    lock = Lock()

    def loop() -> Never:
        while True:
            # Because of lock, _fetch_batch could be inlined into wrapper,
            # and dispatch to thread pool could be done from there,
            # thus allowing usage of scalable ThreadPool
            # TODO: implement above
            with lock:  # Ensurance that none worker steals tasks from other
                batch = _fetch_batch(q, batch_size, timeout)
            if batch:
                _batch_invoke(func, batch)
            else:
                sleep(0.001)

    for _ in range(workers):
        Thread(target=loop, daemon=True).start()
    return q


def streaming[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | None = None,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | BatchFn[T, R]:
    """Delay start of computation to until batch is collected.

    Accepts two timeouts (in seconds):
    - `timeout` is a time to wait till the batch is full, i.e. latency.
    - `pool_timeout` is time to wait for results.

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
    q = _start_fetch_compute(func, workers, batch_size, timeout)

    def wrapper(items: Sequence[T]) -> Sequence[R]:
        fs = {Future(): item for item in items}  # type: ignore[var-annotated]
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
        if isinstance(obj := _gather(fs), BaseException):
            with hide_frame:
                raise obj
        return obj

    # TODO: if func is instance method - recreate wrapper per instance
    # TODO: find how to distinguish between
    # TODO:  not yet bound method and plain function
    # TODO:  maybe implement __get__ on wrapper
    return update_wrapper(wrapper, func)


def _gather[R](fs: Iterable[AnyFuture[R]]) -> list[R] | BaseException:
    cancel: CancelledError | None = None
    errors: dict[BaseException, None] = {}
    results: list[R] = []
    for f in fs:
        if f.cancelled():
            cancel = CancelledError()
        elif exc := f.exception():
            errors[exc] = None
        else:
            results.append(f.result())

    match list(errors):
        case []:
            return cancel or results
        case [err]:
            return err
        case errs:
            msg = 'Got multiple exceptions'
            if all(isinstance(e, Exception) for e in errs):
                return ExceptionGroup(msg, errs)  # type: ignore[type-var]
            return BaseExceptionGroup(msg, errs)
