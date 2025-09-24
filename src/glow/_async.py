__all__ = ['amap', 'amap_dict', 'astarmap', 'azip']

import asyncio
from asyncio import Queue, Task
from collections import deque
from collections.abc import (
    AsyncIterator,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import suppress
from functools import partial
from typing import TypeGuard, cast, overload

from ._dev import hide_frame
from ._futures import adispatch
from ._types import (
    ABatchDecorator,
    ABatchFn,
    AnyFuture,
    AnyIterable,
    AnyIterator,
    Coro,
)

type _Job[T, R] = tuple[T, AnyFuture[R]]


async def amap_dict[K, T1, T2](
    func: Callable[[T1], Coro[T2]],
    obj: Mapping[K, T1],
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
    func: Callable[..., Coro[R]],
    /,
    *iterables: AnyIterable,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
    """Async version of map(func, *iterables).

    Make an iterator that computes the function using arguments from
    each of the iterables. Stops when the shortest iterable is exhausted.

    For extra options, see `astarmap`.
    """
    it = zip(*iterables) if _all_sync_iters(iterables) else azip(*iterables)
    return astarmap(func, it, limit=limit, unordered=unordered)


async def astarmap[*Ts, R](
    func: Callable[[*Ts], Coro[R]],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
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

    async with asyncio.TaskGroup() as tg:
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
    ts: AnyIterator[Task[T]], limit: int
) -> AsyncIterator[T]:
    """Fetch and run async tasks.

    Runs exactly `limit` tasks simultaneously (less in the end of iteration).
    Order of results is arbitrary.
    """
    todo = set[Task[T]]()
    done = Queue[Task[T]]()

    def _todo_to_done(t: Task[T]) -> None:
        todo.discard(t)
        done.put_nowait(t)

    while True:
        # Prefill task buffer
        while len(todo) + done.qsize() < limit and (
            t := (
                next(ts, None)
                if isinstance(ts, Iterator)
                else await anext(ts, None)
            )
        ):
            # optimization: Immediately put to done if the task is
            # already done (e.g. if the coro was able to complete eagerly),
            # and skip scheduling a done callback
            if t.done():
                done.put_nowait(t)
            else:
                todo.add(t)
                t.add_done_callback(_todo_to_done)

        # No more tasks to do and nothing more to schedule
        if not todo and done.empty():
            return

        # Wait till any task succeed
        yield (await done.get()).result()

        # Pop tasks happened to also be DONE (after line above)
        while not done.empty():
            yield done.get_nowait().result()


async def _iter_results[T](
    ts: AnyIterator[Task[T]], limit: int
) -> AsyncIterator[T]:
    """Fetch and run async tasks.

    Runs up to `limit` tasks simultaneously (less in the end of iteration).
    Order of results is preserved.
    """
    todo = deque[Task[T]]()
    while True:
        # Prefill task buffer
        while len(todo) < limit and (
            t := (
                next(ts, None)
                if isinstance(ts, Iterator)
                else await anext(ts, None)
            )
        ):
            todo.append(t)
        if not todo:  # No more tasks to do and nothing more to schedule
            return

        # Forcefully block first task, while it's awaited,
        # others in `todo` are also running, because they are `asyncio.Task`.
        # So after this some of tasks from `todo` are also done.
        yield await todo.popleft()

        # Pop tasks happened to also be DONE (after line above)
        while todo and todo[0].done():
            yield todo.popleft().result()


async def azip(*iterables: AnyIterable) -> AsyncIterator[tuple]:
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
    iterables: Collection[AnyIterable],
) -> TypeGuard[Collection[Iterable]]:
    return all(isinstance(it, Iterable) for it in iterables)


async def _wrapgen[T](it: Iterable[T]) -> AsyncIterator[T]:
    for x in it:
        yield x


@overload
def astreaming(
    *, batch_size: int | None = ..., timeout: float = ...
) -> ABatchDecorator: ...


@overload
def astreaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
) -> ABatchFn[T, R]: ...


def astreaming[T, R](
    fn: ABatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | None = None,
    timeout: float = 0.1,
) -> ABatchFn[T, R] | ABatchDecorator:
    """Compute on `timeout` or if batch is collected.

    `timeout` (in seconds) is a time to wait till the batch is full,
    i.e. latency.

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

    assert batch_size is None or batch_size >= 1
    assert timeout > 0

    buf: list[_Job[T, R]] = []
    deadline = float('-inf')
    not_last = asyncio.Event()
    lock = asyncio.Lock()
    ncalls = 0

    async def wrapper(items: Sequence[T]) -> list[R]:
        nonlocal ncalls, deadline
        if not items:
            return []

        # There's another handling call with tail, wake it up
        if not ncalls and buf:
            not_last.set()

        ncalls += 1
        fs: list[asyncio.Future[R]] = []
        try:
            for x in items:
                f = asyncio.Future[R]()
                fs.append(f)
                buf.append((x, f))

                if len(buf) == 1:  # Got first job, reset deadline
                    deadline = asyncio.get_running_loop().time() + timeout

                # Full batch, dispatch
                if batch_size is not None and len(buf) == batch_size:
                    batch, buf[:] = buf[:], []
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
