__all__ = ['RwLock', 'amap', 'amap_dict', 'astarmap', 'azip']

import asyncio
from asyncio import CancelledError, Event, Future, Lock, Queue, Task, TaskGroup
from collections import deque
from collections.abc import AsyncGenerator, Iterable, Mapping, MutableSet
from contextlib import asynccontextmanager, suppress
from functools import partial
from typing import Literal, Self, TypeGuard, cast, overload
from weakref import finalize

from ._dev import hide_frame
from ._futures import (
    ABatchDecorator,
    ABatchFn,
    ABatchFnRv,
    AJob,
    PsABatchDecorator,
    UsableSize,
    adispatch,
    get_usable_size,
)
from ._types import ACallable, AnyIterable, ASCallable, Get, QueueShutdownError


async def amap_dict[K, T, T2](
    func: ACallable[[T], T2],
    obj: Mapping[K, T],
    /,
    *,
    limit: int,
) -> dict[K, T2]:
    """Asynchronously apply `func` to each value in a mapping.

    For extra options, see astarmap, which is used under hood.
    """
    aiter_values = amap(func, obj.values(), limit=limit)
    values = [v async for v in aiter_values]
    return dict(zip(obj.keys(), values, strict=True))


def amap[R](
    func: ACallable[..., R],
    /,
    *iterables: AnyIterable,
    limit: int,
    unordered: bool = False,
) -> AsyncGenerator[R]:
    """Async version of map(func, *iterables).

    Make an iterator that computes the function using arguments from
    each of the iterables. Stops when the shortest iterable is exhausted.

    For extra options, see `astarmap`.
    """
    it = zip(*iterables) if _all_sync_iters(iterables) else azip(*iterables)
    return astarmap(func, it, limit=limit, unordered=unordered)


async def astarmap[*Ts, R](
    func: ASCallable[*Ts, R],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = False,
) -> AsyncGenerator[R]:
    """Async version of itertools.starmap(fn, iterable).

    Return an iterator whose values are returned from the function evaluated
    with an argument tuple taken from the given sequence.

    Options:
    - limit - Maximum number of simultaneously running tasks.
    - unordered - Set to get yield results as soon as they are ready.
    """
    assert callable(func)

    # optimization: Plain loop if concurrency is unnecessary
    if limit <= 1:
        if isinstance(iterable, Iterable):
            for args in iterable:
                yield await func(*args)
        else:
            async for args in iterable:
                yield await func(*args)
        return

    async with TaskGroup() as tg:
        ts = (
            (tg.create_task(func(*args)) for args in iterable)
            if isinstance(iterable, Iterable)
            else (tg.create_task(func(*args)) async for args in iterable)
        )

        it = (
            _iter_results_unordered(ts, limit=limit)
            if unordered
            else _iter_results(ts, limit=limit)
        )

        async for x in it:
            yield x


async def _iter_results_unordered[T](
    ts: AnyIterable[Task[T]], limit: int
) -> AsyncGenerator[T]:
    """Fetch and run async tasks.

    Runs exactly `limit` tasks simultaneously (less in the end of iteration).
    Order of results is arbitrary.
    """
    it = iter(ts) if isinstance(ts, Iterable) else aiter(ts)
    todo = set[Task[T]]()
    done_q = Queue[Task[T]]()

    def _todo_to_done(t: Task[T]) -> None:
        todo.discard(t)
        done_q.put_nowait(t)

    while True:
        # Prefill task buffer
        while len(todo) + done_q.qsize() < limit and (
            t := (
                next(it, None)
                if isinstance(it, Iterable)
                else await anext(it, None)
            )
        ):
            # optimization: Immediately put to done if the task is
            # already done (e.g. if the coro was able to complete eagerly),
            # and skip scheduling a done callback
            if t.done():
                done_q.put_nowait(t)
            else:
                todo.add(t)
                t.add_done_callback(_todo_to_done)

        # No more tasks to do and nothing more to schedule
        if not todo and done_q.empty():
            return

        # Wait till any task succeed
        yield (await done_q.get()).result()

        # Pop tasks happened to also be DONE (after line above)
        while not done_q.empty():
            yield done_q.get_nowait().result()


async def _iter_results[T](
    ts: AnyIterable[Task[T]], limit: int
) -> AsyncGenerator[T]:
    """Fetch and run async tasks.

    Runs up to `limit` tasks simultaneously (less in the end of iteration).
    Order of results is preserved.
    """
    it = iter(ts) if isinstance(ts, Iterable) else aiter(ts)
    pending = deque[Task[T]]()
    while True:
        # Prefill task buffer
        while len(pending) < limit and (
            t := (
                next(it, None)
                if isinstance(it, Iterable)
                else await anext(it, None)
            )
        ):
            pending.append(t)
        if not pending:  # No more tasks to do and nothing more to schedule
            return

        # Also allows other tasks in `pending` to run and become done.
        # Because they are tasks.
        # To make coroutine run it should be awaited.
        yield await pending.popleft()

        # Pop tasks happened to also be DONE (after line above)
        while pending and pending[0].done():
            yield pending.popleft().result()


async def azip(*iterables: AnyIterable) -> AsyncGenerator[tuple]:
    if _all_sync_iters(iterables):
        for x in zip(*iterables):
            yield x
        return

    aiters = (
        _wrapgen(it) if isinstance(it, Iterable) else aiter(it)
        for it in iterables
    )
    while True:
        try:
            ret = await asyncio.gather(*(anext(ait) for ait in aiters))
        except StopAsyncIteration:
            return
        else:
            yield tuple(ret)


def _all_sync_iters(
    iterables: tuple[AnyIterable, ...],
) -> TypeGuard[tuple[Iterable, ...]]:
    return all(isinstance(it, Iterable) for it in iterables)


async def _wrapgen[T](it: Iterable[T]) -> AsyncGenerator[T]:
    for x in it:
        yield x


@overload
def astreaming(
    *, batch_size: int = ..., timeout: float = ...
) -> ABatchDecorator: ...
@overload
def astreaming[T](
    *, batch_size: UsableSize[T], timeout: float = ...
) -> PsABatchDecorator[T]: ...
@overload
def astreaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
) -> ABatchFnRv[T, R]: ...


def astreaming[T, R](  # noqa: C901
    fn: ABatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
) -> ABatchFnRv[T, R] | PsABatchDecorator[T] | ABatchDecorator:
    """Compute on `timeout` or if batch is collected.

    `timeout` (in seconds) is a time to wait till the batch is full,
    i.e. latency.
    Also if `batch_size` is 0, only timeout is used.

    Uses ideas from
    - https://github.com/ShannonAI/service-streamer
    - https://github.com/leon0707/batch_processor
    - ray.serve.batch
      https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py

    Note: currently supports only functions and bound methods.

    Implementation details:
    - any caller enqueues jobs and starts waiting
    """
    if fn is None:
        deco = partial(astreaming, batch_size=batch_size, timeout=timeout)
        return cast('ABatchDecorator', deco)

    if not callable(batch_size):
        batch_size = partial(get_usable_size, batch_size)
    assert timeout > 0

    buf: list[AJob[T, R]] = []
    deadline = float('-inf')
    not_last = Event()
    lock = Lock()
    ncalls = 0

    async def wrapper(items: Iterable[T]) -> list[R]:
        items = list(items)
        nonlocal ncalls, deadline
        if not items:
            return []

        # There's another handling call with tail, wake it up
        if not ncalls and buf:
            not_last.set()

        ncalls += 1
        fs: list[Future[R]] = []
        try:
            for x in items:
                f = Future[R]()
                fs.append(f)
                buf.append((x, f))

                if len(buf) == 1:  # Got first job, reset deadline
                    deadline = asyncio.get_running_loop().time() + timeout

                usable = batch_size([x for x, _ in buf])
                if not usable:
                    continue

                # Full batch, dispatch
                if usable < len(buf):  # Restart from last append
                    deadline = asyncio.get_running_loop().time() + timeout
                batch, buf[:] = buf[:usable], buf[usable:]
                async with lock:
                    await adispatch(fn, *batch)
        finally:
            ncalls -= 1

        if not ncalls and buf:  # Was last call, wait for another
            not_last.clear()

            notified = False
            with suppress(TimeoutError):
                async with asyncio.timeout_at(deadline):
                    notified = await not_last.wait()

            if not notified:
                batch, buf[:] = buf[:], []
                async with lock:
                    await adispatch(fn, *batch)

        with hide_frame:
            return await asyncio.gather(*fs)

    return wrapper


# ----------------------------- read/write guard -----------------------------


class RwLock:
    """Guard code from concurrent writes.

    Reads are not limited.
    When write is issued, new reads are delayed until write is finished.
    """

    def __init__(self) -> None:
        self._num_reads = 0
        self._readable = Event()
        self._readable.set()
        self._writable = Event()
        self._writable.set()

    @asynccontextmanager
    async def read(self) -> AsyncGenerator[None]:
        await self._readable.wait()
        self._writable.clear()
        self._num_reads += 1
        try:
            yield
        finally:
            self._num_reads -= 1
            if self._num_reads == 0:
                self._writable.set()

    @asynccontextmanager
    async def write(self) -> AsyncGenerator[None]:
        self._readable.clear()  # Stop new READs
        try:
            await self._writable.wait()  # Wait for all READs or single WRITE
            self._writable.clear()  # Only single WRITE is allowed
            try:
                yield
            finally:
                self._writable.set()
        finally:
            self._readable.set()


# --------------------------- multi-consumer queue ---------------------------


class MulticastQueue[T]:
    """Single producer, multiple consumer queue.

    Each consumer gets each message put by producer.
    Late joined consumer gets all messages from the beginning.

    Usage:

        mq = MulticastQueue()
        async def worker():
            with mq:  # or just mq.close() after last mq.put
                for x in range(3):
                    mq.put(x)

        t = asyncio.create_task(worker())
        assert [x async for x in mq] == [0, 1, 2]
        assert [x async for x in mq] == [0, 1, 2]

    """

    def __init__(self) -> None:
        self._buf: list[T] = []
        self._putters = set[Future[None]]()
        self._getters = set[Future[None]]()
        self._state: Literal['running', 'closed', 'terminated'] = 'running'
        self._nsubs = 0

    # ------------------------------- consumer -------------------------------

    async def __aiter__(self) -> AsyncGenerator[T]:
        """Subscribe to queue and iterate over its items."""
        # NOTE: finally doesn't work unless `aclose` is manually called,
        # or anything is `await`ed (even via `asyncio.sleep(0.001)`)
        with self.subscribe() as it:
            async for x in it:
                yield x

    def subscribe(self) -> '_MulticastQueueIterator[T]':
        """Subscribe to queue to get items from its beginning."""
        self._nsubs += 1

        def unsubsribe() -> None:
            self._nsubs -= 1
            if self._state in ('terminated', 'closed'):  # Already stopped
                return
            if self._nsubs:  # Is not the last subscriber
                return
            self._state = 'terminated'
            if self._getters:
                raise RuntimeError(
                    f'{len(self._getters)} non subscribed getters'
                )
            _cancel_all(self._putters, msg='No waiters to store values for')

        return _MulticastQueueIterator(self, unsubsribe)

    async def get(self, idx: int) -> T:
        if idx >= len(self._buf):
            if self._state == 'terminated':  # Closed before finish
                raise QueueShutdownError

            if self._state == 'closed':  # Finished
                raise IndexError

            await _step(
                None,
                wakeup=self._putters,
                wait_for=self._getters,
                cancel_with='Get was interrupted',
            )

        return self._buf[idx]

    # ------------------------------- producer -------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Close queue to mark a successful end of production."""
        if self._state == 'running':
            self._state = 'closed'
            if self._putters:
                raise RuntimeError(
                    'Cannot close queue while there are '
                    f'{len(self._putters)} active writes'
                )
            _cancel_all(self._getters, msg='Queue is finalized')

    async def put(self, value: T) -> None:
        """Put new value to queue."""
        if self._state in ('closed', 'terminated'):
            raise QueueShutdownError

        self._buf.append(value)
        await _step(
            None,
            wakeup=self._getters,
            wait_for=self._putters,
            cancel_with='Put was interrupted',
        )


class _MulticastQueueIterator[T]:
    def __init__(self, mq: MulticastQueue[T], finalizer: Get[None]) -> None:
        self._mq = mq
        self._pos = 0
        self.close = finalize(self, finalizer)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._pos}: ...)'

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            value = await self._mq.get(self._pos)
        except (IndexError, CancelledError) as exc:
            raise StopAsyncIteration from exc
        else:
            self._pos += 1
            return value


async def _step[T](
    value: T,
    /,
    *,
    wakeup: Iterable[Future[T]],
    wait_for: MutableSet[Future[T]],
    cancel_with: str | None = None,
) -> T:
    _wakeup(wakeup, value)
    return await _wait_for(wait_for, cancel_with=cancel_with)


def _wakeup[T](fs: Iterable[Future[T]], value: T) -> None:
    for f in fs:
        if not f.done():
            f.set_result(value)  # Release blocked


async def _wait_for[T](
    fs: MutableSet[Future[T]], cancel_with: str | None = None
) -> T:
    f = Future[T]()
    fs.add(f)
    try:
        return await f  # Acquire to block
    finally:
        f.cancel(cancel_with)
        fs.discard(f)


def _cancel_all(waiters: Iterable[Future], msg: str | None = None) -> None:
    for f in waiters:
        f.cancel(msg)
