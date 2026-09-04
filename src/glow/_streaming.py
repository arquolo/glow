__all__ = ['streaming']

import asyncio
import concurrent.futures as cf
import inspect
import threading
from collections.abc import Iterable
from functools import partial, update_wrapper
from logging import getLogger
from queue import Empty, SimpleQueue
from threading import Thread
from time import monotonic, sleep
from typing import Never, cast, overload

from ._dev import hide_frame
from ._futures import (
    ABatchFn,
    ABatchFnRv,
    AJob,
    BatchDecorator,
    BatchFn,
    BatchFnRv,
    Job,
    PsBatchDecorator,
    UsableSize,
    adispatch,
    dispatch,
    fs_to_results,
    get_usable_size,
)
from ._locking import q_get

_debug1 = partial(getLogger(__name__).debug, stacklevel=2)


@overload
def streaming(
    *,
    batch_size: int = ...,
    timeout: float = ...,
    pool_timeout: float | None = ...,
    workers: int = ...,
) -> BatchDecorator: ...
@overload
def streaming[T](
    *,
    batch_size: UsableSize[T],
    timeout: float = ...,
    pool_timeout: float | None = ...,
    workers: int = ...,
) -> PsBatchDecorator[T]: ...
@overload
def streaming[T, R](
    fn: BatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    pool_timeout: float | None = ...,
    workers: int = ...,
) -> BatchFnRv[T, R]: ...
@overload
def streaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    pool_timeout: float | None = ...,
) -> ABatchFnRv[T, R]: ...


def streaming[T, R](  # noqa: C901
    fn: BatchFn[T, R] | ABatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float | None = None,
) -> BatchFnRv[T, R] | ABatchFnRv[T, R] | PsBatchDecorator[T] | BatchDecorator:
    """Delay start of computation until batch is collected.

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
    - constantly keeps alive N workers (for sync case)
    - any caller enqueues jobs and starts waiting
    - on any failure during waiting caller cancels all jobs it submitted
    - single worker at a time fetches jobs from shared queue, resolves them,
      and notifies all waiters
    """
    if fn is None:
        deco = partial(
            streaming,
            batch_size=batch_size,
            timeout=timeout,
            pool_timeout=pool_timeout,
            workers=workers,
        )
        return cast('BatchDecorator', deco)

    assert callable(fn)
    if not callable(batch_size):
        batch_size = partial(get_usable_size, batch_size)
    assert timeout > 0
    assert workers >= 1

    if inspect.iscoroutinefunction(fn):
        workers = 1
        afn: ABatchFn = fn
        semlock = asyncio.Semaphore(workers)
        ast = _Astream[T, R](batch_size, timeout)

        async def awrapper(items: Iterable[T]) -> list[R]:
            items = list(items)
            if not items:
                return []

            with ast as fs:
                for x in items:
                    if batch := ast.enqueue(fs, x):
                        async with semlock:
                            await adispatch(afn, *batch)

            if batch := await ast.resolve():
                async with semlock:
                    await adispatch(afn, *batch)

            # NOTE: raises only first exception for multiple dead `f` calls
            with hide_frame:
                async with asyncio.timeout(pool_timeout):
                    return await asyncio.gather(*fs)

        return update_wrapper(awrapper, afn)

    st = _Stream(cast('BatchFn[T, R]', fn), batch_size, timeout, workers)

    def wrapper(items: Iterable[T]) -> list[R]:
        fs = {cf.Future[R](): item for item in items}
        try:
            st.enqueue(fs)  # Schedule tasks
            dnd = cf.wait(fs, pool_timeout, return_when='FIRST_EXCEPTION')

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

    # TODO:
    # Recreate wrapper per instance if func is instance method.
    # Find how to distinguish between not yet bound method
    # and plain function, maybe implement __get__ on wrapper.
    return update_wrapper(wrapper, fn)


class _Stream[T, R]:
    def __init__(
        self,
        func: BatchFn[T, R],
        usable_size: UsableSize[T],
        timeout: float,
        workers: int,
    ) -> None:
        # TODO: Use scalable ThreadPool.
        # Track count of active dispatches and scale workers accordingly
        self._func = func
        self._usable_size = usable_size
        self._timeout = timeout

        self._q = SimpleQueue[Job[T, R]]()
        self._lock = threading.Lock()
        self._jobs: list[Job[T, R]] = []
        self._deadline = float('-inf')

        self._run_lock = threading.Lock()  # acquired = started
        self._workers = workers

    def enqueue(self, fs: dict[cf.Future[R], T]) -> None:
        if self._run_lock.acquire(blocking=False):  # Start if not yet started
            for _ in range(self._workers):
                # TODO: scale thread count depending on pool load
                Thread(target=self._batchify, daemon=True).start()

        for f, x in fs.items():
            self._q.put((x, f))  # Schedule task

    def _batchify(self) -> Never:
        while True:
            with self._lock:
                batch = self._next_batch()
            batch = [x for x in batch if x[1].set_running_or_notify_cancel()]
            if batch:
                dispatch(self._func, *batch)
            else:
                sleep(0.001)

    def _next_batch(self) -> list[Job[T, R]]:
        if not self._jobs:  # Wait indefinitely till the first item
            self._jobs[:] = [q_get(self._q)]
            self._deadline = monotonic() + self._timeout

        while not (usable := self._usable_size([x for x, _ in self._jobs])):
            rem = self._deadline - monotonic()
            try:
                j = self._q.get(timeout=max(rem, 0) or None, block=rem > 0)
            except Empty:
                usable = len(self._jobs)
                _debug1(f'worker timed out {self._timeout:.3f}s - qd {usable}')
                break
            else:
                self._jobs.append(j)

        if usable < len(self._jobs):  # Some jobs would remain
            self._deadline = monotonic() + self._timeout

        # Dispatch batch
        jobs, self._jobs = self._jobs[:usable], self._jobs[usable:]
        return jobs


class _Astream[T, R]:
    def __init__(self, usable_size: UsableSize, timeout: float) -> None:
        self._usable_size = usable_size
        self._timeout = timeout

        self._ncalls = 0
        self._not_last = asyncio.Event()
        self._jobs: list[AJob[T, R]] = []
        self._deadline = float('-inf')

    def __enter__(self) -> list[asyncio.Future[R]]:
        # There's another handling call with tail, wake it up
        if not self._ncalls and self._jobs:
            self._not_last.set()

        self._ncalls += 1
        return []

    def __exit__(self, *_) -> None:
        self._ncalls -= 1

    def enqueue(self, fs: list[asyncio.Future[R]], x: T) -> list[AJob[T, R]]:
        f = asyncio.Future[R]()
        fs.append(f)
        self._jobs.append((x, f))

        usable = self._usable_size([x for x, _ in self._jobs])
        if (
            # Got first job...
            len(self._jobs) == 1
            # ...or batch is about to dispatch
            or (usable and usable < len(self._jobs))
        ):
            # Reset deadline
            self._deadline = asyncio.get_running_loop().time() + self._timeout

        if usable:  # Do dispatch
            batch, self._jobs[:] = self._jobs[:usable], self._jobs[usable:]
            return batch
        return []

    async def resolve(self) -> list[AJob[T, R]]:
        if self._ncalls or not self._jobs:
            return []

        # Was last call, wait for another
        self._not_last.clear()
        try:
            async with asyncio.timeout_at(self._deadline):
                await self._not_last.wait()
        except TimeoutError:
            batch, self._jobs[:] = self._jobs[:], []
            return batch
        else:
            return []
