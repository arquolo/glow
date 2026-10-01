import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Generator, Iterable
from weakref import WeakValueDictionary

import numpy as np
import pytest

from glow import azip, chunked, ichunked, windowed


@pytest.mark.parametrize(
    ('arg', 'expected'),
    [
        (range(5), [range(0, 3), range(1, 4), range(2, 5)]),
        (iter(range(5)), [(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
    ],
)
def test_windowed(arg, expected) -> None:
    it = windowed(arg, 3)
    assert [*it] == expected


@pytest.mark.parametrize('fn', [chunked, windowed])
@pytest.mark.parametrize(
    'source',
    [[], [1, 2], (), (1, 2), '', 'ab', b'', b'ab', range(0), range(2)],
)
def test_short_sequence(fn, source) -> None:
    result = list(fn(source, 3))
    assert result == ([source] if source else [])
    for item in result:
        assert type(item) is type(source)


@pytest.mark.parametrize('fn', [chunked, windowed])
@pytest.mark.parametrize('factory', [deque, iter])
@pytest.mark.parametrize(
    ('items', 'expected'),
    [([], []), ([1], [(1,)]), ([1, 2], [(1, 2)]), ([1, 2, 3], [(1, 2, 3)])],
)
def test_short_iterable(fn, factory, items, expected) -> None:
    assert list(fn(factory(items), 3)) == expected


@pytest.mark.parametrize(
    ('arg', 'expected'),
    [
        (range(5), [range(0, 3), range(3, 5)]),
        (iter(range(5)), [(0, 1, 2), (3, 4)]),
    ],
)
def test_chunked(arg, expected) -> None:
    it = chunked(arg, 3)
    assert [*it] == expected


def test_ichunked() -> None:
    it = ichunked(range(5), 3)
    assert len([*it]) == 2

    it = ichunked(range(5), 3)
    assert [tuple(c) for c in it] == [(0, 1, 2), (3, 4)]

    it = ichunked(range(5), 3)
    assert [tuple(c) for c in [*it]] == [(0, 1, 2), (3, 4)]


def _generator(
    n: int, refs: WeakValueDictionary[int, int]
) -> Generator[tuple[int, int]]:
    heap: dict[int, int] = {}
    for i, x in enumerate(np.random.rand(n, 4)):
        refs[i] = heap[i] = x
        del x
        # Careful not to keep reference to x
        yield heap.popitem()


def test_windowed_refs() -> None:
    refs = WeakValueDictionary[int, int]()
    assert not refs

    it = _generator(10, refs)
    for c in windowed(it, 3):
        assert len(refs) <= 3
        del c
    assert not refs


def test_chunked_refs() -> None:
    refs = WeakValueDictionary[int, int]()
    assert not refs

    it = _generator(10, refs)
    for c in chunked(it, 3):
        assert len(refs) <= 3
        del c
    assert not refs


def test_ichunked_refs() -> None:
    refs = WeakValueDictionary[int, int]()
    assert not refs

    it = _generator(10, refs)
    for c in ichunked(it, 3):
        for x in c:
            assert len(refs) <= 1
            del x
    assert not refs

    it = _generator(10, refs)
    chunks = [*ichunked(it, 3)]
    assert len(refs) == 10
    for i, x in zip(range(10, 0, -1), (x for c in chunks for x in c)):
        assert len(refs) <= i
        del x
    assert not refs


class _AsyncItems[T]:
    def __init__(self, items: Iterable[T]) -> None:
        self.items = items
        self.consumed: list[T] = []

    async def __aiter__(self) -> AsyncGenerator[T]:
        for item in self.items:
            self.consumed.append(item)
            yield item


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
@pytest.mark.parametrize('length', [0, 1, 2, 5, 6])
@pytest.mark.parametrize('size', [1, 2, 3, 7])
async def test_async_matches_sync(fn, length, size) -> None:
    source = _AsyncItems(range(length))
    assert [x async for x in fn(source, size)] == list(
        fn(iter(range(length)), size)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
@pytest.mark.parametrize(
    ('items', 'expected'),
    [
        ([], []),
        ([1], [(1,)]),
        ([1, 2], [(1, 2)]),
        ([1, 2, 3], [(1, 2, 3)]),
    ],
)
async def test_short_async_iterable(fn, items, expected) -> None:
    assert [x async for x in fn(_AsyncItems(items), 3)] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('fn', 'ret1', 'ret2'),
    [
        (chunked, (0, 1, 2), (3, 4, 5)),
        (windowed, (0, 1, 2), (1, 2, 3)),
    ],
)
async def test_lazy_consumption(fn, ret1, ret2) -> None:
    source = _AsyncItems(range(10))
    result = fn(source, 3)
    assert source.consumed == []

    assert await anext(result) == ret1
    assert source.consumed == [*ret1]

    assert await anext(result) == ret2
    assert source.consumed == list(range(ret2[-1] + 1))

    await result.aclose()


@pytest.mark.parametrize(
    ('fn', 'size'),
    [
        (chunked, 0),
        (chunked, -1),
        (windowed, 0),
        (windowed, -1),
    ],
)
def test_invalid_size(fn, size) -> None:
    source = _AsyncItems(range(5))
    with pytest.raises(ValueError):
        fn(source, size)
    assert source.consumed == []


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
async def test_source_error(fn):
    async def source() -> AsyncGenerator[int]:
        yield 1
        raise RuntimeError('source failed')

    result = fn(source(), 2)
    with pytest.raises(RuntimeError, match='source failed'):
        await anext(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('factory', [list, tuple, deque, iter, _AsyncItems])
@pytest.mark.parametrize(
    ('fn', 'expected'),
    [
        (chunked, [(0, 1, 2), (3, 4)]),
        (windowed, [(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
    ],
)
async def test_multiple_results(fn, factory, expected) -> None:
    result = fn(factory(range(5)), 3)
    if factory is _AsyncItems:
        actual = [item async for item in result]
    else:
        actual = [tuple(item) for item in result]
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
async def test_source_error_after_result(fn) -> None:
    error = RuntimeError('source failed')

    async def source() -> AsyncGenerator[int]:
        yield 0
        yield 1
        yield 2
        raise error

    result = fn(source(), 3)
    assert await anext(result) == (0, 1, 2)
    with pytest.raises(RuntimeError, match='source failed') as exc:
        await anext(result)
    assert exc.value is error
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
async def test_cancel_pending_next(fn) -> None:
    waiting = asyncio.Event()
    blocker = asyncio.Event()
    closed = asyncio.Event()

    async def source() -> AsyncGenerator[int]:
        try:
            yield 0
            waiting.set()
            await blocker.wait()
            yield 1
        finally:
            closed.set()

    result = fn(source(), 3)
    task = asyncio.create_task(anext(result))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        with pytest.raises(StopAsyncIteration):
            await anext(result)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await result.aclose()


class _TrackedItem:
    pass


async def _tracked_items(refs) -> AsyncGenerator[_TrackedItem]:
    heap = {}
    for i in range(10):
        heap[i] = _TrackedItem()
        refs[i] = heap[i]
        yield heap.pop(i)


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
async def test_async_releases_consumed_items(fn) -> None:
    refs = WeakValueDictionary()
    async for batch in fn(_tracked_items(refs), 3):
        assert len(refs) <= 3
        del batch
    assert not refs


@pytest.mark.asyncio
@pytest.mark.parametrize('fn', [chunked, windowed])
async def test_async_close_releases_buffer(fn) -> None:
    refs = WeakValueDictionary()
    source = _tracked_items(refs)
    result = fn(source, 3)
    try:
        batch = await anext(result)
        assert len(refs) == 3
        del batch
        await result.aclose()
        assert not refs
    finally:
        await result.aclose()
        await source.aclose()


@pytest.mark.parametrize('fn', [chunked, windowed])
@pytest.mark.parametrize('factory', [list, deque, iter])
@pytest.mark.parametrize('size', [0, -1])
def test_invalid_size_sync(fn, factory, size) -> None:
    source = factory(range(5))
    with pytest.raises(ValueError):
        fn(source, size)
    assert list(source) == list(range(5))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['error', 'exhausted', 'cancel'])
async def test_azip_cleans_pending_tasks(failure):
    started = asyncio.Event()
    closed = asyncio.Event()
    blocker = asyncio.Event()

    async def blocked():
        try:
            started.set()
            await blocker.wait()
            yield 1
        finally:
            closed.set()

    async def other():
        await started.wait()
        if failure == 'error':
            raise ValueError('source failed')
        if failure == 'cancel':
            await blocker.wait()
        return
        yield

    result = azip(blocked(), other())
    task = asyncio.create_task(anext(result))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if failure == 'cancel':
            task.cancel()
        expected = {
            'error': ValueError,
            'exhausted': StopAsyncIteration,
            'cancel': asyncio.CancelledError,
        }[failure]
        with pytest.raises(expected):
            await asyncio.wait_for(task, timeout=1)
        assert closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await result.aclose()


@pytest.mark.asyncio
async def test_azip_sync_anext_error_cleans_created_future():
    pending = asyncio.get_running_loop().create_future()

    class Source:
        def __aiter__(self):
            return self

        def __anext__(self):
            return pending

    class Broken(Source):
        def __anext__(self):
            raise ValueError('immediate failure')

    with pytest.raises(ValueError, match='immediate failure'):
        await anext(azip(Source(), Broken()))
    assert pending.cancelled()
