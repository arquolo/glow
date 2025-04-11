__all__ = ['amap', 'astarmap', 'azip']

import asyncio
from asyncio import Queue, Task
from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
)
from typing import Any, TypeGuard

type _AnyIterable[T] = AsyncIterable[T] | Iterable[T]
type _AnyIterator[T] = AsyncIterator[T] | Iterator[T]


def amap[
    R
](
    func: Callable[..., Coroutine[Any, Any, R]],
    /,
    *iterables: _AnyIterable,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
    it = zip(*iterables) if _all_sync_iters(iterables) else azip(*iterables)
    return astarmap(func, it, limit=limit, unordered=unordered)


async def astarmap[
    *Ts, R
](
    func: Callable[[*Ts], Coroutine[Any, Any, R]],
    iterable: _AnyIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
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


async def _iter_results_unordered[
    T
](ts: _AnyIterator[Task[T]], limit: int) -> AsyncIterator[T]:
    """
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


async def _iter_results[
    T
](ts: _AnyIterator[Task[T]], limit: int) -> AsyncIterator[T]:
    """
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


async def azip(*iterables: _AnyIterable) -> AsyncIterator[tuple]:
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
    iterables: Collection[_AnyIterable],
) -> TypeGuard[Collection[Iterable]]:
    return all(isinstance(it, Iterable) for it in iterables)


async def _wrapgen[T](it: Iterable[T]) -> AsyncIterator[T]:
    for x in it:
        yield x
