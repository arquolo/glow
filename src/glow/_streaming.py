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

from ._dev import hide_frame, name_object
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
from ._locking import q_get, set_future

_logger = getLogger(__name__)


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
        afn = cast('ABatchFn[T, R]', fn)
        ast = _Astream(afn, batch_size, timeout, workers)
        enqueue_tasks = set[asyncio.Task[None]]()

        async def awrapper(items: Iterable[T]) -> list[R]:
            fs = {asyncio.Future[R](): item for item in items}
            if not fs:
                return []

            task = asyncio.create_task(ast.enqueue(fs))
            enqueue_tasks.add(task)
            task.add_done_callback(enqueue_tasks.discard)
            try:
                async with asyncio.timeout(pool_timeout):
                    with hide_frame:
                        return await asyncio.gather(*fs)
            finally:
                for f in fs:
                    f.cancel()

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
        rs, err = fs_to_results(enumerate(fs))
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
        latency: float,
        workers: int,
    ) -> None:
        # TODO: Use scalable ThreadPool.
        # Track count of active dispatches and scale workers accordingly
        self._func = func
        self._usable_size = usable_size
        self._latency = latency

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
            self._deadline = monotonic() + self._latency

        while True:
            try:
                usable = self._usable_size([x for x, _ in self._jobs])
            except BaseException as exc:
                if not self._jobs:
                    raise
                log_msg = (
                    f'{name_object(self._usable_size)} on '
                    f'{name_object(self._func)} failed with {exc!r}. '
                    'Using batch 1'
                )
                # Could be traced only up to `_thread.start_joinable_thread`,
                # not func() call, thus no `stacklevel=...`
                _logger.exception(log_msg)
                usable = 1

            if usable:
                break

            rem = self._deadline - monotonic()
            try:
                j = self._q.get(timeout=max(rem, 0) or None, block=rem > 0)
            except Empty:
                usable = len(self._jobs)
                log_msg = (
                    f'Worker timed out for {name_object(self._func)}, '
                    f'{self._latency:.3f}s - qd {usable}'
                )
                # Could be traced only up to `_thread.start_joinable_thread`,
                # not func() call, thus no `stacklevel=...`
                _logger.debug(log_msg)
                break
            else:
                self._jobs.append(j)

        if usable < len(self._jobs):  # Some jobs would remain
            self._deadline = monotonic() + self._latency

        # Dispatch batch
        jobs, self._jobs = self._jobs[:usable], self._jobs[usable:]
        return jobs


class _Astream[T, R]:
    def __init__(
        self,
        func: ABatchFn[T, R],
        usable_size: UsableSize,
        latency: float,
        workers: int,
    ) -> None:
        self._func = func
        self._usable_size = usable_size
        self._latency = latency

        self._ncalls = 0
        self._not_last = asyncio.Event()
        self._jobs: list[AJob[T, R]] = []
        self._deadline = float('-inf')
        self._run_lock = asyncio.Semaphore(workers)

    async def enqueue(self, fs: dict[asyncio.Future[R], T]) -> None:
        try:
            if not self._ncalls and self._jobs:  # Wake up tail handler
                self._not_last.set()
            self._ncalls += 1
            try:
                for f, x in fs.items():
                    self._jobs.append((x, f))  # Enqueue

                    if batch := self._next_batch():
                        async with self._run_lock:
                            await adispatch(self._func, *batch)
            finally:
                self._ncalls -= 1

            if batch := await self._resolve():
                async with self._run_lock:
                    await adispatch(self._func, *batch)

        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                for f in fs:
                    f.cancel()
                raise
            for f in fs:
                set_future(f, exc)

    def _next_batch(self) -> list[AJob[T, R]]:
        try:
            usable = self._usable_size([x for x, _ in self._jobs])
        except BaseException as exc:
            if not self._jobs:
                raise
            log_msg = (
                f'{name_object(self._usable_size)} on '
                f'{name_object(self._func)} failed with {exc!r}. Using batch 1'
            )
            # Could be traced only up to `_thread.start_joinable_thread`,
            # not func() call, thus no `stacklevel=...`
            _logger.exception(log_msg)
            usable = 1

        if (
            len(self._jobs) == 1  # Got first job...
            or 0 < usable < len(self._jobs)  # ...or batch is about to dispatch
        ):
            # Reset deadline
            self._deadline = asyncio.get_running_loop().time() + self._latency

        if usable:  # Do dispatch
            batch, self._jobs[:] = self._jobs[:usable], self._jobs[usable:]
            return batch
        return []

    async def _resolve(self) -> list[AJob[T, R]]:
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
