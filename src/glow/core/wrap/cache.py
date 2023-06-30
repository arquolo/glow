from __future__ import annotations

__all__ = ['memoize']

import argparse
import asyncio
import concurrent.futures as cf
import enum
import functools
from collections import Counter
from collections.abc import (Callable, Hashable, Iterable, KeysView,
                             MutableMapping)
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import RLock
from typing import (Any, ClassVar, Generic, Literal, NamedTuple, SupportsInt,
                    TypeVar, cast)
from weakref import WeakValueDictionary

from .._repr import si_bin
from .._sizeof import sizeof
from .reusable import make_loop
from .util import make_key


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_BatchedFn = TypeVar('_BatchedFn', bound=Callable[[list], Iterable])
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]
_empty = _Empty.token


def _unit_size(x):
    return 1


@dataclass(repr=False)
class _Node(Generic[_T]):
    __slots__ = ('value', 'size')
    value: _T
    size: int

    def __repr__(self) -> str:
        return repr(self.value)


class Stats(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__ = Counter()
        self.__dict__.update(**kwargs)

    def __getattr__(self, name: str) -> int:
        return self.__dict__[name]


class _IStore(Generic[_T]):
    def __len__(self) -> int:
        raise NotImplementedError

    def store_clear(self) -> None:
        raise NotImplementedError

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        raise NotImplementedError

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        raise NotImplementedError

    def can_shrink_for(self, size: int) -> bool:
        raise NotImplementedError


@dataclass(repr=False)
class _InitializedStore:
    capacity: int
    size_fn: Callable[[object], int]
    size: int = field(default=0, init=False)
    stats: Stats = field(default_factory=Stats, init=False)


@dataclass(repr=False)
class _DictMixin(_InitializedStore, _IStore[_T]):
    lock: RLock = field(default_factory=RLock, init=False)

    def clear(self):
        with self.lock:
            self.store_clear()
            self.size = 0

    def keys(self) -> KeysView:
        raise NotImplementedError

    def __getitem__(self, key: Hashable) -> _T | _Empty:
        with self.lock:
            if node := self.store_get(key):
                self.stats.hits += 1
                return node.value
        return _empty

    def __setitem__(self, key: Hashable, value: _T) -> None:
        with self.lock:
            self.stats.misses += 1
            size = int(self.size_fn(value))
            if (self.capacity < 0  # is unbound
                    or self.size + size <= self.capacity  # has free place
                    or (size < self.capacity and self.can_shrink_for(size))):
                self.store_set(key, _Node(value, size))
                self.size += size


@dataclass(repr=False)
class _ReprMixin(_InitializedStore, _IStore[_T]):
    refs: ClassVar[MutableMapping[int, _ReprMixin]] = WeakValueDictionary()

    def __post_init__(self) -> None:
        self.refs[id(self)] = self

    def __repr__(self) -> str:
        args = [
            f'items={len(self)}',
            f'used={si_bin(self.size)}',
            f'total={si_bin(self.capacity)}',
        ]
        if any(vars(self.stats).values()):
            args.append(f'stats={self.stats}')
        return f'{type(self).__name__}({", ".join(args)})'

    @classmethod
    def status(cls) -> str:
        return '\n'.join(
            f'{id_:x}: {value!r}' for id_, value in sorted(cls.refs.items()))


@dataclass(repr=False)
class _Store(_ReprMixin[_T], _DictMixin[_T]):
    store: dict[Hashable, _Node[_T]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.store)

    def keys(self) -> KeysView:
        return self.store.keys()

    def store_clear(self) -> None:
        self.store.clear()

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        return self.store.get(key)

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        self.store[key] = node


class _HeapCache(_Store[_T]):
    def can_shrink_for(self, size: int) -> bool:
        return False


class _LruCache(_Store[_T]):
    drop_recent = False

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        if node := self.store.pop(key, None):
            self.store[key] = node
            return node
        return None

    def can_shrink_for(self, size: int) -> bool:
        size = 0
        while self.store and self.size + size > self.capacity:
            if self.drop_recent:
                self.size -= self.store.popitem()[1].size
            else:
                self.size -= self.store.pop(next(iter(self.store))).size
            self.stats.dropped += 1
        return True


class _MruCache(_LruCache[_T]):
    drop_recent = True


# -------------------------------- wrapping --------------------------------


def _memoize(cache: _DictMixin, key_fn: _KeyFn, fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = key_fn(*args, **kwargs)

        if (value := cache[key]) is not _empty:
            return value

        # NOTE: fn() is not within lock
        cache[key] = value = fn(*args, **kwargs)
        return value

    wrapper.cache = cache  # type: ignore[attr-defined]
    return cast(_F, functools.update_wrapper(wrapper, fn))


# ----------------------- wrapper with batching support ----------------------


class _Job(NamedTuple):
    token: Any
    future: asyncio.Future | cf.Future


def _dispatch(
    fn: Callable[[list], Iterable],
    evict: Callable[[Hashable], object],
    queue: dict[Hashable, _Job],
):
    jobs = {**queue}
    queue.clear()

    try:
        *values, = fn([job.token for job in jobs.values()])
        assert len(values) == len(jobs)

        for job, value in zip(jobs.values(), values):
            job.future.set_result(value)

    except BaseException as exc:  # noqa: BLE001
        for key, job in jobs.items():
            evict(key)
            job.future.set_exception(exc)


def _memoize_batched_aio(key_fn: _KeyFn, fn: _BatchedFn) -> _BatchedFn:
    assert callable(fn)
    futs: dict[Hashable, asyncio.Future] = {}
    queue: dict[Hashable, _Job] = {}
    loop = make_loop()

    def _load(token) -> asyncio.Future:
        key = key_fn(token)
        if result := futs.get(key):
            return result

        loop = asyncio.get_running_loop()
        futs[key] = future = loop.create_future()
        queue[key] = _Job(token, future)
        if len(queue) == 1:
            loop.call_soon(_dispatch, fn, futs.pop, queue)

        return future

    async def _load_many(tokens: Iterable) -> tuple:
        return await asyncio.gather(*map(_load, tokens))

    def wrapper(tokens: Iterable) -> tuple:
        coro = _load_many(tokens)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    wrapper.stage = futs  # type: ignore[attr-defined]
    return cast(_BatchedFn, functools.update_wrapper(wrapper, fn))


def _memoize_batched(cache: _DictMixin, key_fn: _KeyFn,
                     fn: _BatchedFn) -> _BatchedFn:
    assert callable(fn)
    lock = RLock()
    futs = WeakValueDictionary[Hashable, cf.Future]()
    queue: dict[Hashable, _Job] = {}

    def _load(stack: ExitStack, key: Hashable, token: object) -> cf.Future:
        with lock:
            if result := futs.get(key):
                return result

            futs[key] = future = cf.Future()  # type: ignore[var-annotated]
            queue[key] = _Job(token, future)
            if len(queue) == 1:
                stack.callback(_dispatch, fn, futs.pop, queue)

        return future

    def wrapper(tokens: Iterable) -> list:
        keyed_tokens = [(key_fn(t), t) for t in tokens]

        # Try to hit
        misses = {}
        hits = {}
        with cache.lock:
            for k, t in dict(keyed_tokens).items():
                if (r := cache[k]) is not _empty:
                    hits[k] = r
                else:
                    misses[k] = t

        futs: dict[Hashable, cf.Future] = {}
        with ExitStack() as stack:
            futs |= {k: _load(stack, k, t) for k, t in misses.items()}
        cf.wait(futs.values(), return_when='FIRST_EXCEPTION')

        # Process misses
        with cache.lock:
            for k, f in futs.items():
                hits[k] = cache[k] = f.result()
        return [hits[k] for k, _ in keyed_tokens]

    wrapper.cache = cache  # type: ignore[attr-defined]
    wrapper.stage = futs  # type: ignore[attr-defined]
    return cast(_BatchedFn, functools.update_wrapper(wrapper, fn))


# ----------------------------- factory wrappers -----------------------------


def memoize(
    capacity: SupportsInt | None,
    *,
    batched: bool = False,
    policy: _Policy = 'raw',
    key_fn: _KeyFn = make_key,
    bytesize: bool = True,
) -> Callable[[_F], _F] | Callable[[_BatchedFn], _BatchedFn]:
    """Returns dict-cache decorator.

    Parameters:
    - capacity - max size in bytes if `bytesize` is set, otherwise objects.
      Cache is unbound if None set.
    - policy - eviction policy, either "raw" (no eviction), or "lru"
      (evict oldest), or "mru" (evict most recent).
    - bytesize - if set limits bytes, not objects.
    """
    capacity = int(capacity) if capacity is not None else -1
    if capacity == 0:
        return lambda fn: fn

    caches: dict[str, type[_Store]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    size_fn = sizeof if bytesize else _unit_size

    if (cache_cls := caches.get(policy)) is not None:
        if capacity < 0:
            cache_cls = _HeapCache
        cache = cache_cls(capacity, size_fn)
        if batched:
            return functools.partial(_memoize_batched, cache, key_fn)
        return functools.partial(_memoize, cache, key_fn)
    raise ValueError(f'Unknown policy: "{policy}". '
                     f'Only "{set(caches)}" are available')
