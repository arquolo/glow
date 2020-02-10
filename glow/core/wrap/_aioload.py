__all__ = ('batched', 'batched_async')

import asyncio
import concurrent.futures as cf
import contextlib
import functools
import threading
from dataclasses import dataclass, field
from typing import (Dict, Hashable, Iterable, List, NamedTuple, TypeVar, Union,
                    cast)

from typing_extensions import Protocol

from .reusable import make_loop

_T_co = TypeVar('_T_co', covariant=True)


class _BatchFn(Protocol[_T_co]):
    def __call__(self, __keys: Iterable[Hashable]) -> Iterable[_T_co]:
        ...


class _Entry(NamedTuple):
    key: Hashable
    future: Union[asyncio.Future, cf.Future]


def _dispatch(fn: _BatchFn, cache: Dict[Hashable, object],
              queue: List[_Entry]) -> None:
    queue[:], queue = [], queue[:]

    keys = [l.key for l in queue]
    try:
        values = *fn(keys),
        assert len(values) == len(keys)

        for l, value in zip(queue, values):
            l.future.set_result(value)

    except BaseException as exc:
        for l in queue:
            cache.pop(l.key)
            l.future.set_exception(exc)


_B = TypeVar('_B', bound=_BatchFn)

# ------------------------------ asyncio-based ------------------------------


@dataclass
class _Loader:
    fn: _BatchFn
    _cache: Dict[Hashable, asyncio.Future] = field(default_factory=dict)
    _queue: List[_Entry] = field(default_factory=list)

    async def load_many(self, keys: Iterable[Hashable]) -> Iterable:
        return await asyncio.gather(*map(self.load, keys))

    def load(self, key: Hashable) -> asyncio.Future:
        result = self._cache.get(key)
        if result is not None:
            return result

        loop = asyncio.get_running_loop()

        self._cache[key] = future = loop.create_future()
        self._queue.append(_Entry(key, future))
        if len(self._queue) == 1:
            loop.call_soon(_dispatch, self.fn, self._cache, self._queue)

        return future


def batched_async(fn: _B) -> _B:
    assert callable(fn)
    ul = _Loader(fn)

    def wrapper(keys: Iterable[Hashable]) -> Iterable:
        coro = ul.load_many(keys)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    loop = make_loop()
    wrapper.cache = ul._cache  # type: ignore
    return cast(_B, functools.update_wrapper(wrapper, fn))


# ------------------------- concurrent-future-based -------------------------


def batched(fn: _B) -> _B:
    """Applies `fn` to not-seen-before items in batch"""
    assert callable(fn)
    lock = threading.RLock()
    cache: Dict[Hashable, cf.Future] = {}
    queue: List[_Entry] = []

    def wrapper(keys: Iterable[Hashable]) -> Iterable:
        with contextlib.ExitStack() as stack:
            futs = [_resolve(stack, key) for key in keys]
        return [fut.result() for fut in futs]

    def _resolve(stack: contextlib.ExitStack, key: Hashable) -> cf.Future:
        with lock:
            result = cache.get(key)
            if result is not None:
                return result

            cache[key] = future = cf.Future()  # type: ignore
            queue.append(_Entry(key, future))
            if len(queue) == 1:
                stack.callback(_dispatch, fn, cache, queue)

        return future

    wrapper.cache = cache  # type: ignore
    return cast(_B, functools.update_wrapper(wrapper, fn))