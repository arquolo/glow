import asyncio
import random
import time
from threading import Event, Thread
from unittest.mock import MagicMock

import pytest

import glow


def _lrange(*args: int) -> list[int]:
    return list(range(*args))


def test_bad_len():
    @glow.memoize(5, batched=True)
    def fn(xs):
        return []

    with pytest.raises(RuntimeError):
        fn([None])
    assert not dict(fn.cache)


@pytest.mark.asyncio
async def test_bad_len_async():
    @glow.memoize(5, batched=True)
    async def fn(xs):
        return []

    with pytest.raises(RuntimeError):
        await fn([None])
    assert not dict(fn.cache)


def test_seq():
    load = MagicMock(side_effect=lambda x: x)

    @glow.memoize(20, batched=True)
    def load_batch(xs):
        return [load(x) for x in xs]

    for _ in range(5):
        numbers = random.choices(range(20), k=5)
        results = load_batch(numbers)
        assert numbers == results

    called_with = (x for (x,), _ in load.call_args_list)
    assert sorted(called_with) == sorted(load_batch.cache)


@pytest.mark.asyncio
async def test_seq_async():
    load = MagicMock(side_effect=lambda x: x)

    @glow.memoize(20, batched=True)
    async def load_batch(xs):
        await asyncio.sleep(0)
        return [load(x) for x in xs]

    for _ in range(5):
        numbers = random.choices(range(20), k=5)
        results = await load_batch(numbers)
        assert numbers == results

    called_with = (x for (x,), _ in load.call_args_list)
    assert sorted(called_with) == sorted(load_batch.cache)


@pytest.mark.parametrize('max_workers', [2, 4] * 10)
def test_map(max_workers):
    @glow.memoize(100, batched=True)
    def load_batch(xs):
        return [*xs]

    numbers = random.choices(range(100), k=100)
    nchunks = glow.chunked(numbers, 5)
    rchunks = glow.map_n(load_batch, nchunks, max_workers=max_workers)
    assert [r for c in rchunks for r in c] == numbers


@pytest.mark.asyncio
@pytest.mark.parametrize('max_workers', [2, 4] * 10)
async def test_amap(max_workers):
    @glow.memoize(100, batched=True)
    async def load_batch(xs):
        await asyncio.sleep(0)
        return [*xs]

    numbers = random.choices(range(100), k=100)
    nchunks = glow.chunked(numbers, 5)
    rchunks = [
        x async for x in glow.amap(load_batch, nchunks, limit=max_workers)
    ]
    assert [r for c in rchunks for r in c] == numbers


def test_concurrent():
    ev = Event()

    @glow.memoize(100, batched=True)
    def fn(xs):
        ev.set()
        time.sleep(0.1)
        return xs

    t = Thread(target=lambda: fn(_lrange(0, 10)))
    t.start()
    ev.wait()

    x1 = fn(_lrange(5, 15))
    t.join()

    assert x1 == _lrange(5, 15)
    assert dict(fn.cache).keys() == set(_lrange(15))


@pytest.mark.asyncio
async def test_concurrent_async():
    ev = asyncio.Event()

    @glow.memoize(100, batched=True)
    async def fn(xs):
        ev.set()
        await asyncio.sleep(0.1)
        return xs

    t = asyncio.create_task(fn(_lrange(0, 10)))
    await ev.wait()

    x1 = await fn(_lrange(5, 15))
    assert x1 == _lrange(5, 15)

    x2 = await t
    assert x2 == _lrange(0, 10)

    assert dict(fn.cache).keys() == set(_lrange(15))


@pytest.mark.asyncio
async def test_concurrent_async_interrupted():
    ev = asyncio.Event()

    @glow.memoize(100, batched=True)
    async def fn(xs):
        ev.set()
        await asyncio.sleep(0.2)
        return xs

    t = asyncio.create_task(fn(_lrange(10)))
    await ev.wait()

    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.1):
            await fn(_lrange(5, 15))

    await t
    assert dict(fn.cache).keys() == set(_lrange(10))


@pytest.mark.parametrize('workers', [0, 1, 3])
@pytest.mark.parametrize('unordered', [False, True])
def test_sync_map(workers, unordered):
    def fn(x):
        time.sleep(0.01 / (1 + x))
        return x

    n = 10
    xs = list(range(n))

    items = list(glow.map_n(fn, xs, max_workers=workers, unordered=unordered))
    if unordered:
        items = sorted(items)
    assert items == xs

    items = list(
        glow.starmap_n(
            fn, ((x,) for x in xs), max_workers=workers, unordered=unordered
        )
    )
    if unordered:
        items = sorted(items)
    assert items == xs


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', [0, 1, 3])
@pytest.mark.parametrize('unordered', [False, True])
async def test_async_map(limit, unordered):
    async def fn(x):
        await asyncio.sleep(0.01 / (1 + x))
        return x

    n = 10
    xs = list(range(n))

    items = [
        x async for x in glow.amap(fn, xs, limit=limit, unordered=unordered)
    ]
    if unordered:
        items = sorted(items)
    assert items == xs

    items = [
        x
        async for x in glow.astarmap(
            fn, ((x,) for x in xs), limit=limit, unordered=unordered
        )
    ]
    if unordered:
        items = sorted(items)
    assert items == xs


def _usable_items(xs):
    if sum(xs) < 7:
        return 0
    return max(1, len(xs) - 1)


def test_stream_1():
    calls = []

    @glow.streaming(batch_size=3)
    def fn(xs):
        calls.append(xs)
        return xs

    fn(range(5))
    assert calls == [[0, 1, 2], [3, 4]]


def test_stream_2():
    calls = []

    @glow.streaming(batch_size=3)
    def fn(xs):
        print(xs)
        calls.append(xs)
        time.sleep(0.01)
        return xs

    t1 = Thread(target=fn, args=(range(5),), daemon=True)
    t2 = Thread(target=fn, args=(range(5, 10),), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert all(len(cs) <= 3 for cs in calls)
    assert sorted(c for cs in calls for c in cs) == list(range(10))


def test_stream_dyn_lim():
    calls = []

    @glow.streaming(batch_size=_usable_items)
    def fn(xs):
        print(xs)
        calls.append(xs)
        time.sleep(0.01)
        return xs

    fn(range(5))
    assert calls == [[0, 1, 2, 3], [4]]

    calls = []
    t1 = Thread(target=fn, args=(range(5),), daemon=True)
    t2 = Thread(target=fn, args=(range(5, 8),), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert all(sum(cs) <= 7 for cs in calls)
    assert sorted(c for cs in calls for c in cs) == list(range(8))


@pytest.mark.asyncio
async def test_astream_1():
    calls = []

    @glow.astreaming(batch_size=3)
    async def fn(xs):
        calls.append(xs)
        return xs

    await fn(range(5))
    assert calls == [[0, 1, 2], [3, 4]]


@pytest.mark.asyncio
async def test_astream_2():
    calls = []

    @glow.astreaming(batch_size=3)
    async def fn(xs):
        print(xs)
        calls.append(xs)
        await asyncio.sleep(0.01)
        return xs

    await asyncio.gather(fn(range(5)), fn(range(5, 10)))
    assert all(len(cs) <= 3 for cs in calls)
    assert sorted(c for cs in calls for c in cs) == list(range(10))


@pytest.mark.asyncio
async def test_astream_dyn_lim():
    calls = []

    @glow.astreaming(batch_size=_usable_items)
    async def fn(xs):
        print(xs)
        calls.append(xs)
        await asyncio.sleep(0.01)
        return xs

    await fn(range(5))
    assert calls == [[0, 1, 2, 3], [4]]

    calls = []
    await asyncio.gather(fn(range(5)), fn(range(5, 8)))
    assert calls == [[0, 1, 2, 3], [4], [5], [6], [7]]
