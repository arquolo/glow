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
from concurrent.futures import Future, wait
from functools import partial, update_wrapper
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import NoReturn
from warnings import warn

from loguru import logger

from ._cache import memoize

type _BatchFn[T, R] = Callable[[list[T]], Iterable[R]]

_PATIENCE = 0.01


def threadlocal[T](
    fn: Callable[..., T], /, *args: object, **kwargs: object
) -> Callable[[], T]:
    """Thread-local singleton factory, mimics `functools.partial`"""
    local_ = threading.local()

    def wrapper() -> T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)


def call_once[T](fn: Callable[[], T], /) -> Callable[[], T]:
    """Makes callable a singleton.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions"""
    warn(
        'Deprecated. Use `@memoize()` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize()(fn)


def shared_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """Merges duplicate parallel invocations of callable to a single one.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions"""
    warn(
        'Deprecated. Use `@memoize(0)` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize(0)(fn)


def weak_memoize[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """Preserves each result of each call until they are garbage collected."""
    warn(
        'Deprecated. Use `@memoize(0)` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize(0)(fn)


# ----------------------------- batch collation ------------------------------


def _fetch_batch[T](
    q: SimpleQueue[T], batch_size: int, timeout: float
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
    while len(batch) < batch_size and (waittime := endtime - monotonic()) > 0:
        try:
            batch.append(q.get(timeout=waittime))
        except Empty:
            break
    return batch


def _batch_invoke[T, R](
    func: _BatchFn[T, R], batch: Sequence[tuple[Future[R], T]]
) -> None:
    batch = [(f, x) for f, x in batch if f.set_running_or_notify_cancel()]
    if not batch:
        return

    try:
        results = [*func([x for _, x in batch])]
        if len(results) != len(batch):
            raise RuntimeError(  # noqa: TRY301
                'Input batch size is not equal to output: '
                f'{len(results)} != {len(batch)}'
            )

    except BaseException as exc:  # noqa: BLE001
        for f, _ in batch:
            f.set_exception(exc)

    else:
        for (f, _), r in zip(batch, results):
            f.set_result(r)


def _start_fetch_compute(func, workers, batch_size, timeout):
    q = SimpleQueue()  # type: ignore[var-annotated]
    lock = Lock()

    def loop() -> NoReturn:
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


def streaming(
    func=None, /, *, batch_size, timeout=0.1, workers=1, pool_timeout=20.0
):
    """
    Delays start of computation to until batch is collected.
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
        return partial(
            streaming,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout,
        )

    assert callable(func)
    assert workers >= 1
    q = _start_fetch_compute(func, workers, batch_size, timeout)

    def wrapper(items):
        fs = {Future(): item for item in items}  # type: ignore[var-annotated]
        try:
            for f_x in fs.items():
                q.put(f_x)  # Schedule task
            dnd = wait(fs, pool_timeout, return_when='FIRST_EXCEPTION')

        finally:  # Cancel all not-yet-running tasks, we're beyond deadline
            for f in fs:
                f.cancel()

        if dnd.not_done:  # Some tasks timed out
            del dnd, fs  # ? Break reference cycle
            raise TimeoutError
        return [f.result() for f in fs]  # Cannot time out - all are done

    # TODO: if func is instance method - recreate wrapper per instance
    # TODO: find how to distinguish between
    # TODO:  not yet bound method and plain function
    # TODO:  maybe implement __get__ on wrapper
    return update_wrapper(wrapper, func)
