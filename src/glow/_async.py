__all__ = ['amap', 'astarmap', 'azip']

import asyncio
from asyncio import Task
from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
)
from itertools import starmap
from typing import Any, TypeGuard

type _Coro[T] = Coroutine[Any, Any, T]


def amap[
    R
](
    func: Callable[..., _Coro[R]],
    /,
    *iterables: Iterable | AsyncIterable,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
    return astarmap(func, azip(*iterables), limit=limit, unordered=unordered)


async def astarmap[
    *Ts, R
](
    func: Callable[[*Ts], _Coro[R]],
    iterable: Iterable[tuple[*Ts]] | AsyncIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = False,
) -> AsyncIterator[R]:
    assert callable(func)
    if isinstance(iterable, Iterable):
        aws = _wrapgen(starmap(func, iterable))
    else:
        aws = (func(*args) async for args in iterable)

    async with asyncio.TaskGroup() as tg:
        ts = (tg.create_task(aw) async for aw in aws)
        it = (
            _iter_results_unordered(ts, limit=limit)
            if unordered
            else _iter_results(ts, limit=limit)
        )
        async for x in it:
            yield x


async def _iter_results_unordered[
    T
](ts: AsyncIterator[Task[T]], limit: int) -> AsyncIterator[T]:
    """
    Runs exactly `limit` tasks simultaneously (less in the end of iteration).
    Order of results is arbitrary.
    """
    todo = set[Task[T]]()
    while True:
        # Prefill task buffer
        while len(todo) < limit and (t := await anext(ts, None)):
            todo.add(t)
        if not todo:
            return

        # Pop done tasks
        done, todo = await asyncio.wait(
            todo, return_when=asyncio.FIRST_COMPLETED
        )
        while done:
            yield await done.pop()
        if not todo:
            return


async def _iter_results[
    T
](ts: AsyncIterator[Task[T]], limit: int) -> AsyncIterator[T]:
    """
    Runs up to `limit` tasks simultaneously (less in the end of iteration).
    Order of results is preserved.
    """
    todo = deque[Task[T]]()
    while True:
        # Prefill task buffer
        while len(todo) < limit and (t := await anext(ts, None)):
            todo.append(t)

        # Pop done tasks
        await asyncio.wait(todo, return_when=asyncio.FIRST_COMPLETED)
        while todo[0].done():
            yield await todo.popleft()

        if not todo:
            return


async def azip(*iterables: Iterable | AsyncIterable) -> AsyncIterator[tuple]:
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
    iterables: tuple[Iterable | AsyncIterable, ...]
) -> TypeGuard[tuple[Iterable, ...]]:
    return all(isinstance(it, Iterable) for it in iterables)


async def _wrapgen[T](it: Iterable[T]) -> AsyncIterator[T]:
    for x in it:
        yield x
