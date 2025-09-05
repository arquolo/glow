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
