__all__ = ['cache_status', 'memoize']

import asyncio
import concurrent.futures as cf
import functools
from collections.abc import (
    Awaitable,
    Callable,
    MutableMapping,
    Hashable,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
)
from dataclasses import dataclass
from inspect import (
    isasyncgenfunction,
    iscoroutinefunction,
    isgeneratorfunction,
)
from threading import RLock
from time import monotonic
from typing import Final, Protocol, SupportsInt
from weakref import WeakValueDictionary

from ._dev import clone_exc, hide_frame
from ._futures import (
    ABatchFn,
    ABatchFnRv,
    AnyFuture,
    AnyBatchDecorator,
    BatchFn,
    BatchFnRv,
    adispatch,
    dispatch,
    gather_fs,
)
from ._keys import make_key
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import CachePolicy, Decorator, Empty, KeyFn, empty

_inf: Final = float('inf')


@dataclass(repr=False, slots=True)
class _Node[T]:
    value: T
    nbytes: int = 0
    deadline: float = _inf

    def __repr__(self) -> str:
        return repr(self.value)


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    dropped: int = 0

    def __bool__(self) -> bool:
        return any(self.__dict__.values())

    def __repr__(self) -> str:
        fields = ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if v)
        return f'{self.__class__.__name__}({fields})'


# ----------------------------- basic caches ------------------------------


def cache_status() -> str:
    return '\n'.join(
        f'{id_:x}: {value!r}' for id_, value in sorted(_REFS.items())
    )


_REFS = WeakValueDictionary[int, '_Cache']()


class _AbstractCache[T](Protocol):
    def __getitem__(self, key: Hashable, /) -> T | Empty: ...
    def __setitem__(self, key: Hashable, value: T, /) -> None: ...


class _CacheMaker[T](Protocol):
    def __call__(
        self,
        capacity: int,
        capacity_bytes: int,
        ttl: float = ...,
    ) -> _AbstractCache[T]: ...


class _Cache[T]:
    __slots__ = (
        '__weakref__',
        'capacity',
        'capacity_bytes',
        'nbytes',
        'stats',
        'store',
        'ttl',
    )

    def __init__(
        self,
        capacity: int,
        capacity_bytes: int,
        ttl: float = _inf,
        store: dict[Hashable, _Node[T]] | None = None,
    ) -> None:
        self.capacity = capacity
        self.capacity_bytes = capacity_bytes
        self.nbytes = 0
        self.stats = Stats()
        self.store = {} if store is None else store
        self.ttl = ttl

    def __post_init__(self) -> None:
        assert self.capacity != 0
        assert self.capacity_bytes != 0
        _REFS[id(self)] = self

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.store)

    def keys(self) -> KeysView[Hashable]:
        return self.store.keys()

    def clear(self) -> None:
        self.stats.dropped += len(self.store)
        self.store.clear()
        self.nbytes = 0

    def __repr__(self) -> str:
        args = [
            f'items={len(self.store)}',
            f'size={si_bin(self.nbytes)}',
            f'capacity={self.capacity}',
            f'capacity[bytes]={si_bin(self.capacity_bytes)}',
        ]
        if self.stats:
            args.append(f'stats={self.stats}')
        return f'{type(self).__name__}({", ".join(args)})'

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        deadline = self._prune_and_get_new_deadline()

        if node := self.store.pop(key, None):
            self.stats.hits += 1
            node.deadline = deadline
            self.store[key] = node  # move front
            return node.value

        self.stats.misses += 1
        return empty

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        node = self.store.pop(key, None)  # pop before GC to reuse size
        deadline = self._prune_and_get_new_deadline()

        if node:
            node.deadline = deadline
            self.store[key] = node  # move front
        else:
            nbytes = sizeof(value) if self.capacity_bytes > 0 else 0
            node = _Node(value, nbytes, deadline)
            self._maybe_insert(key, node)

    def _maybe_insert(self, key: Hashable, node: _Node[T], /) -> None:
        if (0 < self.capacity <= len(self.store)) or (
            0 < self.capacity_bytes < self.nbytes + node.nbytes
        ):  # no free space
            return
        self.store[key] = node
        self.nbytes += node.nbytes

    def _prune_and_get_new_deadline(self) -> float:
        return _inf


class _TimedCache[T](_Cache[T]):
    """
    Drops items older than `now - ttl`
    """

    def _prune_and_get_new_deadline(self) -> float:
        now = monotonic()
        deadline = now + self.ttl
        while self.store:
            key, node = next(iter(self.store.items()))
            if node.deadline > now:
                return deadline  # reached alive node before free space
            self.store.pop(key)  # dead node, delete
            self.nbytes -= node.nbytes
            self.stats.dropped += 1
        return deadline


class _EvictableCache[T](_Cache[T]):
    """
    Evicts nodes when cache is too large
    """

    def _maybe_insert(self, key: Hashable, node: _Node[T], /) -> None:
        if self.store and len(self.store) == self.capacity:  # no space
            self.nbytes -= self.pop()  # evict
            self.stats.dropped += 1

        if self.capacity_bytes > 0:  # byte-bound cache
            max_self_bytes_to_fit = self.capacity_bytes - node.nbytes
            if max_self_bytes_to_fit < 0:  # cache will never fit this
                return
            while self.store and self.nbytes > max_self_bytes_to_fit:  # evict
                self.nbytes -= self.pop()
                self.stats.dropped += 1

        self.store[key] = node
        self.nbytes += node.nbytes

    def pop(self) -> int:
        raise NotImplementedError


class _LruMixin:
    """Evicts least recently used node when cache is too large."""

    store: MutableMapping[Hashable, _Node]

    def pop(self) -> int:
        """Drop oldest node."""
        return self.store.pop(next(iter(self.store))).nbytes


class _MruMixin:
    """Evicts most recently used node when cache is too large."""

    store: MutableMapping[Hashable, _Node]

    def pop(self) -> int:
        """Drop most recently added node."""
        return self.store.popitem()[1].nbytes


class _LruCache[T](_LruMixin, _EvictableCache[T]):
    pass


class _MruCache[T](_MruMixin, _EvictableCache[T]):
    pass


class _TimedLruCache[T](_LruMixin, _EvictableCache[T], _TimedCache[T]):
    pass


class _TimedMruCache[T](_MruMixin, _EvictableCache[T], _TimedCache[T]):
    pass


# --------------------------------- utilities --------------------------------


class _WeakCache[T]:
    """Retrieve items via weak references from everywhere."""

    def __init__(self) -> None:
        self.alive = WeakValueDictionary[Hashable, T]()

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        return self.alive.get(key, empty)

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        if type(value).__weakrefoffset__:  # Support weak reference.
            self.alive[key] = value


class _StrongCache[T](_WeakCache[T]):
    def __init__(self, cache: _AbstractCache[T]) -> None:
        super().__init__()
        self.cache = cache

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        # Alive and stored items.
        # Called first to update cache stats (i.e. MRU/LRU if any).
        # `cache` has subset of objects from `alive`.
        if (ret := self.cache[key]) is not empty:
            return ret
        # Item could still exist, try reference ...
        return super().__getitem__(key)

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        self.cache[key] = value
        super().__setitem__(key, value)


# --------------------------------- wrapping ---------------------------------


def _result[T](f: cf.Future[T]) -> T:
    if f.cancelled():
        with hide_frame:
            raise cf.CancelledError
    if exc := f.exception():
        with hide_frame:
            raise exc
    return f.result()


def _sync_memoize[**P, R](
    fn: Callable[P, R],
    cache: _AbstractCache[R],
    key_fn: KeyFn[P],
) -> Callable[P, R]:
    futures = WeakValueDictionary[Hashable, cf.Future[R]]()
    lock = RLock()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = key_fn(*args, **kwargs)

        is_owner = False
        with lock:
            if (ret := cache[key]) is not empty:
                return ret

            # ... or it could be computed somewhere else, join there.
            f = futures.get(key)
            if not f:
                futures[key] = f = cf.Future[R]()
                is_owner = True

        # Release lock to allow function to run
        if not is_owner:
            with hide_frame:
                return _result(f)

        try:
            with hide_frame:
                ret = fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(clone_exc(exc))
            with lock:
                futures.pop(key)
            raise
        else:
            f.set_result(ret)
            with lock:
                cache[key] = ret
                futures.pop(key)
            return ret

    return _update_wrapper(wrapper, fn, cache, futures)


def _async_memoize[**P, R](
    fn: Callable[P, Awaitable[R]],
    cache: _AbstractCache[R],
    key_fn: KeyFn[P],
) -> Callable[P, Awaitable[R]]:
    futures = WeakValueDictionary[Hashable, asyncio.Future[R]]()

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = key_fn(*args, **kwargs)

        if (ret := cache[key]) is not empty:
            return ret

        # ... or it could be computed somewhere else, join there.
        if f := futures.get(key):
            with hide_frame:
                return await f
        futures[key] = f = asyncio.Future[R]()

        # NOTE: fn() is not within threading.Lock, thus it's not thread safe
        # NOTE: but it's async-safe because this `await` is only one here.
        try:
            with hide_frame:
                ret = await fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(clone_exc(exc))
            futures.pop(key)
            raise
        else:
            f.set_result(ret)
            cache[key] = ret
            futures.pop(key)
            return ret

    return _update_wrapper(wrapper, fn, cache, futures)


# ----------------------- wrapper with batching support ----------------------


class _BatchedQuery[T, F: AnyFuture, R]:
    def __init__(
        self,
        cache: _AbstractCache[R],
        futures: MutableMapping[Hashable, F],
        new_future: type[F],
        *keyed_tokens: tuple[Hashable, T],
    ) -> None:
        self._keys = [k for k, _ in keyed_tokens]  # All keys with duplicates
        self._done: dict[Hashable, R] = {}

        self.running = set[F]()  # Wait for these
        self.pending: list[tuple[T, F]] = []  # Populate those
        self._futures: dict[Hashable, F] = {}

        for k, t in dict(keyed_tokens).items():
            # If this key is processing right now, wait till its done ...
            if f := futures.get(k):  # ! Protect
                self._futures[k] = f
                self.running.add(f)  # Wait for these

            # ... else check if it's done ...
            elif (r := cache[k]) is not empty:  # ! Protect
                self._done[k] = r

            # ... otherwise schedule a new job.
            else:
                futures[k] = self._futures[k] = f = new_future()  # ! Protect
                self.pending.append((t, f))  # Resolve this manually

    def partial_result(
        self,
    ) -> tuple[dict[Hashable, R], BaseException | None]:
        return gather_fs(self._futures)

    def merge(
        self, stash: Mapping[Hashable, R], cache: _AbstractCache[R]
    ) -> None:
        for k, r in stash.items():
            self._done[k] = cache[k] = r

    # TODO: check whether this necessary
    def cleanup(self, futures: MutableMapping[Hashable, F]) -> None:
        for k in self._futures:  # Force next callers to use cache
            futures.pop(k, None)

    def result(self) -> list[R]:
        return [self._done[k] for k in self._keys]


def _sync_memoize_batched[T, R](
    fn: BatchFn[T, R], cache: _AbstractCache[R], key_fn: KeyFn
) -> BatchFnRv[T, R]:
    futures = WeakValueDictionary[Hashable, cf.Future[R]]()
    lock = RLock()

    def wrapper(tokens: Iterable[T]) -> list[R]:
        keyed_tokens = [(key_fn(t), t) for t in tokens]

        with lock:
            q = _BatchedQuery(cache, futures, cf.Future[R], *keyed_tokens)

        if not q.pending and not q.running:
            return q.result()

        try:
            if q.pending:
                dispatch(fn, *q.pending)
            if q.running:
                cf.wait(q.running)
            stash, err = q.partial_result()
        except:
            with lock:
                q.cleanup(futures)
            raise
        else:
            with lock:
                q.merge(stash, cache)
                q.cleanup(futures)

        if err is None:
            return q.result()
        with hide_frame:
            raise err

    return _update_wrapper(wrapper, fn, cache, futures)


def _async_memoize_batched[T, R](
    fn: ABatchFn[T, R], cache: _AbstractCache[R], key_fn: KeyFn[T]
) -> ABatchFnRv[T, R]:
    futures = WeakValueDictionary[Hashable, asyncio.Future[R]]()

    async def wrapper(tokens: Iterable[T]) -> list[R]:
        keyed_tokens = [(key_fn(t), t) for t in tokens]
        q = _BatchedQuery(cache, futures, asyncio.Future[R], *keyed_tokens)

        if not q.pending and not q.running:
            return q.result()

        try:
            if q.pending:
                await adispatch(fn, *q.pending)
            if q.running:
                await asyncio.wait(q.running)
            stash, err = q.partial_result()
            q.merge(stash, cache)
        finally:
            q.cleanup(futures)

        if err is None:
            return q.result()
        with hide_frame:
            raise err

    return _update_wrapper(wrapper, fn, cache, futures)


def _update_wrapper[F: Callable](
    wrapper: F,
    fn: Callable,
    cache: _AbstractCache,
    futures: MutableMapping,
) -> F:
    wrapper.futures = futures  # type: ignore[attr-defined]
    if isinstance(cache, _WeakCache):
        wrapper.wrefs = cache.alive  # type: ignore[attr-defined]
    if isinstance(cache, _StrongCache):
        wrapper.cache = cache.cache  # type: ignore[attr-defined]
    functools.update_wrapper(wrapper, fn)
    return wrapper


# -------------------------------- decoration --------------------------------


def memoize(
    count: SupportsInt | None = None,
    *,
    nbytes: SupportsInt | None = None,
    batched: bool = False,
    policy: CachePolicy | None = None,
    key_fn: KeyFn = make_key,
    ttl: float | None = None,
) -> Decorator | AnyBatchDecorator:
    """Create caching decorator.

    Parameters:
    - count - max objects to store or None for unbound cache.
    - nbytes - max bytes to store.
    - policy - eviction policy, "lru" (pop oldest), "mru" (pop most recent), or
      None for no eviction. Works only if `count > 0` or `nbytes > 0`.
    - batched - set if callable supports batching.
    - ttl - time to live (in seconds) for time constrained caching

    Uses:
    - @memoize() - unbound cache;
    - @memoize(batched=True) - unbound cache for batched calls;
    - @memoize(<int>, policy=...) - limit cache size by object count;
    - @memoize(nbytes=..., policy=...) - limit cache size by total object size;
    - @memoize(ttl=...) - limit cache size by lifetime of object.
    """
    count = -1 if count is None else int(count)
    nbytes = -1 if nbytes is None else si_bin(int(nbytes))

    # +/+, +/0, +/-, 0/+, 0/0, 0/-, -/+, -/0, -/-
    if (count == 0 and nbytes > 0) or (count > 0 and nbytes == 0):
        raise ValueError(
            'Ambiguity: if one of count/nbytes is 0,'
            f'then other should be 0 or -1. Got: {count} and {nbytes}'
        )
    if count < 0 and nbytes < 0:  # Unbound cache, eviction policy is useless
        policy = None

    # +/+, +/-, 0/0, 0/-, -/+, -/0, -/-
    if count == 0 or nbytes == 0 or (ttl is not None and ttl <= 0):
        # 0/0, 0/-, -/0 (weakrefs only)
        cache = _WeakCache()

    # +/+(count+nbytes), +/-(count), -/+(nbytes), -/-(unbound)
    elif cache_cls := _CACHES.get((policy, bool(ttl))):
        cache = _StrongCache(cache_cls(count, nbytes, ttl or _inf))
    else:
        msg = f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
        raise ValueError(msg)

    def wrap(fn: Callable) -> Callable:
        if isasyncgenfunction(fn) or isgeneratorfunction(fn):
            raise TypeError(f'Generator functions are not supported. Got {fn}')

        if iscoroutinefunction(fn):
            if batched:
                return _async_memoize_batched(fn, cache, key_fn)
            return _async_memoize(fn, cache, key_fn)
        if batched:
            return _sync_memoize_batched(fn, cache, key_fn)
        return _sync_memoize(fn, cache, key_fn)

    return wrap


_CACHES: dict[tuple[CachePolicy | None, bool], _CacheMaker] = {
    (None, False): _Cache,
    ('lru', False): _LruCache,
    ('mru', False): _MruCache,
    (None, True): _TimedCache,
    ('lru', True): _TimedLruCache,
    ('mru', True): _TimedMruCache,
}
