import asyncio
import gc
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, NoReturn
from unittest.mock import MagicMock

import pytest

import glow
import glow._cache as cache_module


class Value:
    pass


def test_memoizes_by_arguments() -> None:
    fn = MagicMock(side_effect=lambda *args, **kwargs: (args, kwargs))
    memoized = glow.memoize()(fn)

    first = memoized(1, flag=True)
    second = memoized(1, flag=True)

    assert first is second
    assert fn.call_count == 1
    assert memoized(2, flag=True) != first
    assert fn.call_count == 2


def test_custom_key_fn() -> None:
    fn = MagicMock(side_effect=lambda value: object())
    memoized = glow.memoize(key_fn=lambda value: value.casefold())(fn)

    assert memoized('Spam') is memoized('SPAM')
    fn.assert_called_once_with('Spam')


@pytest.mark.asyncio
async def test_memoizes_async_function() -> None:
    calls = 0

    @glow.memoize()
    async def fn(value) -> object:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return object()

    first, second = await asyncio.gather(fn(1), fn(1))

    assert first is second
    assert calls == 1
    assert await fn(1) is first
    assert calls == 1


@pytest.mark.asyncio
async def test_async_exception_is_retrieved() -> None:
    loop = asyncio.get_running_loop()
    contexts = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    @glow.memoize()
    async def fn() -> NoReturn:
        raise RuntimeError('boom')

    try:
        with pytest.raises(RuntimeError, match='boom'):
            await fn()
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not contexts


def test_merges_concurrent_calls() -> None:
    barrier = Barrier(2)
    calls = 0

    @glow.memoize()
    def fn(value) -> object:
        nonlocal calls
        calls += 1
        barrier.wait()
        return object()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(fn, 1)
        barrier.wait()
        result = fn(1)

    assert future.result() is result
    assert calls == 1


def test_does_not_cache_exceptions() -> None:
    calls = 0

    @glow.memoize()
    def fn() -> NoReturn:
        nonlocal calls
        calls += 1
        raise RuntimeError('boom')

    for _ in range(2):
        with pytest.raises(RuntimeError, match='boom'):
            fn()

    assert calls == 2
    assert not fn.futures  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ('policy', 'expected_keys'),
    [('lru', {1, 3}), ('mru', {2, 3})],
)
def test_eviction_policy(policy, expected_keys) -> None:
    @glow.memoize(2, policy=policy)
    def fn(value) -> object:
        return object()

    one = fn(1)
    fn(2)
    assert fn(1) is one
    fn(3)

    assert set(fn.cache) == expected_keys  # type: ignore[attr-defined]
    assert fn.cache.stats.dropped == 1  # type: ignore[attr-defined]


def test_ttl(monkeypatch) -> None:
    now = 10.0
    monkeypatch.setattr(cache_module, 'monotonic', lambda: now)

    @glow.memoize(ttl=5)
    def fn() -> object:
        return object()

    first = fn()
    now = 14.0
    assert fn() is first

    now = 20.0
    assert fn() is not first
    assert fn.cache.stats.dropped == 1  # type: ignore[attr-defined]


def test_zero_capacity_keeps_only_weak_references() -> None:
    calls = 0

    @glow.memoize(0)
    def fn():
        nonlocal calls
        calls += 1
        return Value()

    value = fn()
    assert fn() is value
    assert calls == 1

    del value
    gc.collect()

    assert not fn.wrefs  # type: ignore[attr-defined]
    assert isinstance(fn(), Value)
    assert calls == 2


@pytest.mark.parametrize(
    ('count', 'nbytes'),
    [(0, 1), (1, 0)],
)
def test_rejects_ambiguous_capacity(count, nbytes) -> None:
    with pytest.raises(ValueError, match='Ambiguity'):
        glow.memoize(count, nbytes=nbytes)


def test_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match='Unknown cache policy'):
        glow.memoize(1, policy='random')  # type: ignore[call-overload]


def test_rejects_generator_function() -> None:
    def fn() -> Generator[None, Any]:
        yield None

    with pytest.raises(
        TypeError, match='Generator functions are not supported'
    ):
        glow.memoize()(fn)


def test_rejects_async_generator_function() -> None:
    async def fn() -> AsyncGenerator[None, Any]:
        yield None

    with pytest.raises(
        TypeError, match='Generator functions are not supported'
    ):
        glow.memoize()(fn)
