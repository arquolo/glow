__all__ = [
    'cache_status',
    'call_once',
    'coalesce',
    'memoize',
]

import asyncio
import concurrent.futures as cf
import functools
import inspect
from collections.abc import Callable, Hashable, Iterable, Iterator, KeysView
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Final, Protocol, SupportsInt
from warnings import warn
from weakref import WeakValueDictionary

from ._dev import clone_exc, hide_frame
from ._futures import (
    ABatchFn,
    ABatchFnRv,
    AnyFuture,
    BatchFn,
    BatchFnRv,
    adispatch,
    as_maybe,
    dispatch,
    fs_to_results,
)
from ._keys import make_key
from ._locking import maybe_future
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import ACallable, CachePolicy, Empty, Get, KeyFn, empty

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
    def __delitem__(self, key: Hashable, /) -> None: ...


class _CacheMaker[T](Protocol):
    def __call__(
        self, capacity: int, capacity_bytes: int, ttl: float = ...
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

    def __delitem__(self, key: Hashable, /) -> None:
        if node := self.store.pop(key, None):
            self.nbytes -= node.nbytes
            self.stats.dropped += 1
        self._prune_and_get_new_deadline()

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


class _LruCache[T](_EvictableCache[T]):
    """Evicts least recently used node when cache is too large."""

    def pop(self) -> int:
        """Drop oldest node."""
        return self.store.pop(next(iter(self.store))).nbytes


class _MruCache[T](_EvictableCache[T]):
    """Evicts most recently used node when cache is too large."""

    def pop(self) -> int:
        """Drop most recently added node."""
        return self.store.popitem()[1].nbytes


class _TimedLruCache[T](_LruCache, _TimedCache[T]):
    pass


class _TimedMruCache[T](_MruCache, _TimedCache[T]):
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

    def __delitem__(self, key: Hashable, /) -> None:
        self.alive.pop(key, None)


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

    def __delitem__(self, key: Hashable, /) -> None:
        del self.cache[key]
        super().__delitem__(key)


# ----------------------- wrapper with batching support ----------------------


def _prepare_batch[T, F: AnyFuture, R](
    cache: _AbstractCache[R],
    futures: WeakValueDictionary[Hashable, F],
    new_future: type[F],
    *keyed_tokens: tuple[Hashable, T],
) -> tuple[
    list[Hashable],  # keys
    list[tuple[T, F]],  # pending
    set[F],  # running
    dict[Hashable, F],  # futures
    dict[Hashable, R],  # done
]:
    done: dict[Hashable, R] = {}
    running = set[F]()  # Wait for these
    pending: list[tuple[T, F]] = []  # Populate those
    fs: dict[Hashable, F] = {}

    for k, t in dict(keyed_tokens).items():
        # If this key is processing right now, wait till its done ...
        if f := futures.get(k):  # ! Protect
            fs[k] = f
            running.add(f)  # Wait for these

        # ... else check if it's done ...
        elif (r := cache[k]) is not empty:  # ! Protect
            done[k] = r

        # ... otherwise schedule a new job.
        else:
            futures[k] = fs[k] = f = new_future()  # ! Protect
            pending.append((t, f))  # Resolve this manually

    return (
        [k for k, _ in keyed_tokens],  # All keys with duplicates
        pending,
        running,
        fs,
        done,
    )


# -------------------------------- decoration --------------------------------


class memoize:  # noqa: N801
    """Caching decorator.

    Parameters:
    - count - max objects to store or None for unbound cache.
    - nbytes - max bytes to store.
    - policy - eviction policy, "lru" (pop oldest), "mru" (pop most recent), or
      None for no eviction. Works only if `count > 0` or `nbytes > 0`.
    - batched - set if callable supports batching.
    - ttl - time to live (in seconds) for time constrained caching

    Uses:
    - @memoize() - unbound cache;
    - @memoize(0) - cache by weakref only and just merge simulateneous calls;
    - @memoize(batched=True) - unbound cache for batched calls;
    - @memoize(<int>, policy=...) - limit cache size by object count;
    - @memoize(nbytes=..., policy=...) - limit cache size by total object size;
    - @memoize(ttl=...) - limit cache size by lifetime of object.
    - @memoize().drop(fn) - mark function invalidating cache,
      should be key-compatible with main function.

    All functions sharing one `memoize` instance, including functions wrapped
    with `.drop`, must be either synchronous or asynchronous.
    """

    def __init__(
        self,
        count: SupportsInt | None = None,
        *,
        nbytes: SupportsInt | None = None,
        batched: bool = False,
        policy: CachePolicy | None = None,
        key_fn: KeyFn = make_key,
        ttl: float | None = None,
    ) -> None:
        count = -1 if count is None else int(count)
        nbytes = -1 if nbytes is None else si_bin(int(nbytes))

        # +/+, +/0, +/-, 0/+, 0/0, 0/-, -/+, -/0, -/-
        if (count == 0 and nbytes > 0) or (count > 0 and nbytes == 0):
            raise ValueError(
                'Ambiguity: if one of count/nbytes is 0,'
                f'then other should be 0 or -1. Got: {count} and {nbytes}'
            )
        if (
            count < 0 and nbytes < 0
        ):  # Unbound cache, eviction policy is useless
            policy = None

        # +/+, +/-, 0/0, 0/-, -/+, -/0, -/-
        if count == 0 or nbytes == 0 or (ttl is not None and ttl <= 0):
            # 0/0, 0/-, -/0 (weakrefs only)
            self._cache = _WeakCache()

        # +/+(count+nbytes), +/-(count), -/+(nbytes), -/-(unbound)
        elif cache_cls := _CACHES.get((policy, bool(ttl))):
            self._cache = _StrongCache(cache_cls(count, nbytes, ttl or _inf))
        else:
            raise ValueError(
                f'Unknown cache policy: "{policy}". '
                f'Available: "{set(_CACHES)}"'
            )

        self._key_fn = key_fn
        self._batched = batched
        self._lock = RLock()
        self._futures = WeakValueDictionary[Hashable, cf.Future]()
        self._afutures = WeakValueDictionary[Hashable, asyncio.Future]()
        self._is_async: bool | None = None

    def __call__(self, fn: Callable) -> Callable:
        if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn):
            raise TypeError(f'Generator functions are not supported. Got {fn}')

        if inspect.iscoroutinefunction(fn):
            if self._is_async is False:
                raise TypeError('Cannot use sync cache for async function')
            self._is_async = True
        else:
            if self._is_async is True:
                raise TypeError('Cannot use async cache for sync function')
            self._is_async = False

        w = (
            (self._awrap_batched(fn) if self._batched else self._awrap(fn))
            if inspect.iscoroutinefunction(fn)
            else (self._wrap_batched(fn) if self._batched else self._wrap(fn))
        )
        return self._update_wrapper(w, fn)

    def drop(self, fn: Callable) -> Callable:
        if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn):
            raise TypeError(f'Generator functions are not supported. Got {fn}')

        if inspect.iscoroutinefunction(fn):
            if self._is_async is False:
                raise TypeError('Cannot use sync cache for async function')
            self._is_async = True
        else:
            if self._is_async is True:
                raise TypeError('Cannot use async cache for sync function')
            self._is_async = False

        w = (
            (self._adrop_batched(fn) if self._batched else self._adrop(fn))
            if inspect.iscoroutinefunction(fn)
            else (self._drop_batched(fn) if self._batched else self._drop(fn))
        )
        return self._update_wrapper(w, fn)

    def _wrap[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = self._key_fn(*args, **kwargs)

            is_owner = False
            with self._lock:
                if (ret := self._cache[key]) is not empty:
                    return ret

                # ... or it could be computed somewhere else, join there.
                f = self._futures.get(key)
                if not f:
                    self._futures[key] = f = cf.Future[R]()
                    is_owner = True

            # Release lock to allow function to run
            if not is_owner:
                with hide_frame:
                    obj = maybe_future(f)
                    if isinstance(obj, BaseException):
                        raise obj
                return obj[0]

            try:
                with hide_frame:
                    ret = fn(*args, **kwargs)
            except BaseException as exc:
                f.set_exception(clone_exc(exc))
                with self._lock:
                    self._futures.pop(key)
                raise
            else:
                f.set_result(ret)
                with self._lock:
                    self._cache[key] = ret
                    self._futures.pop(key)
                return ret

        return wrapper

    def _awrap[**P, R](self, fn: ACallable[P, R]) -> ACallable[P, R]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = self._key_fn(*args, **kwargs)

            if (ret := self._cache[key]) is not empty:
                return ret

            # ... or it could be computed somewhere else, join there.
            if f := self._afutures.get(key):
                with hide_frame:
                    return await f
            self._afutures[key] = f = asyncio.Future[R]()

            # NOTE: `fn()` isn't thread safe - no threading.Lock,
            # but it's async-safe because this `await` is the only one here.
            try:
                with hide_frame:
                    ret = await fn(*args, **kwargs)
            except BaseException as exc:
                f.set_exception(clone_exc(exc))
                f.exception()  # Mark as retrieved if nobody awaits the future.
                self._afutures.pop(key)
                raise
            else:
                f.set_result(ret)
                self._cache[key] = ret
                self._afutures.pop(key)
                return ret

        return wrapper

    def _wrap_batched[T, R](self, fn: BatchFn[T, R]) -> BatchFnRv[T, R]:
        def wrapper(tokens: Iterable[T]) -> list[R]:
            keyed_tokens = [(self._key_fn(t), t) for t in tokens]

            with self._lock:
                keys, pending, running, futures, done = _prepare_batch(
                    self._cache, self._futures, cf.Future, *keyed_tokens
                )

            if not futures:
                return [done[k] for k in keys]

            if pending:
                dispatch(fn, *pending)
            if running:
                cf.wait(running)
            stash, err = fs_to_results(futures.items())
            with self._lock:
                for k, r in stash.items():
                    done[k] = self._cache[k] = r

            if err is None:
                return [done[k] for k in keys]
            with hide_frame:
                raise err

        return wrapper

    def _awrap_batched[T, R](self, fn: ABatchFn[T, R]) -> ABatchFnRv[T, R]:
        async def wrapper(tokens: Iterable[T]) -> list[R]:
            keyed_tokens = [(self._key_fn(t), t) for t in tokens]
            keys, pending, running, futures, done = _prepare_batch(
                self._cache, self._afutures, asyncio.Future, *keyed_tokens
            )

            if not futures:
                return [done[k] for k in keys]

            if pending:
                await adispatch(fn, *pending)
            if running:
                await asyncio.wait(running)
            stash, err = fs_to_results(futures.items())
            for k, r in stash.items():
                done[k] = self._cache[k] = r

            if err is None:
                return [done[k] for k in keys]
            with hide_frame:
                raise err

        return wrapper

    def _drop[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = self._key_fn(*args, **kwargs)

            with self._lock:  # Check if other call is running
                f = self._futures.get(key)
            try:
                with hide_frame:
                    if f:  # Wait for completion of previous calls
                        maybe_future(f)
                    return fn(*args, **kwargs)
            finally:
                with self._lock:
                    del self._cache[key]

        return wrapper

    def _adrop[**P, R](self, fn: ACallable[P, R]) -> ACallable[P, R]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = self._key_fn(*args, **kwargs)
            try:
                # Wait other calls
                if (f := self._afutures.get(key)) and not f.done():
                    await asyncio.wait([f])
                with hide_frame:
                    return await fn(*args, **kwargs)
            finally:
                del self._cache[key]

        return wrapper

    def _drop_batched[T, R](self, fn: BatchFn[T, R]) -> BatchFnRv[T, R]:
        def wrapper(tokens: Iterable[T]) -> list[R]:
            kts = [(self._key_fn(t), t) for t in tokens]
            if not kts:
                return []

            ukts = dict(kts)
            with self._lock:
                fs = [f for k in ukts if (f := self._futures.get(k))]
            try:
                with hide_frame:
                    cf.wait(fs)
                    obj = as_maybe(fn(list(ukts.values())), len(ukts))
                    if isinstance(obj, list):
                        map_ = dict(zip(ukts, obj))
                        return [map_[k] for k, _ in kts]
                    raise obj
            finally:
                with self._lock:
                    for k in ukts:
                        del self._cache[k]

        return wrapper

    def _adrop_batched[T, R](self, fn: ABatchFn[T, R]) -> ABatchFnRv[T, R]:
        async def wrapper(tokens: Iterable[T]) -> list[R]:
            kts = [(self._key_fn(t), t) for t in tokens]
            if not kts:
                return []

            ukts = dict(kts)
            fs = [f for k in ukts if (f := self._afutures.get(k))]
            try:
                with hide_frame:
                    if fs:
                        await asyncio.wait(fs)
                    obj = as_maybe(await fn(list(ukts.values())), len(ukts))
                    if isinstance(obj, list):
                        map_ = dict(zip(ukts, obj))
                        return [map_[k] for k, _ in kts]
                    raise obj
            finally:
                for k in ukts:
                    del self._cache[k]

        return wrapper

    def _update_wrapper[F: Callable](self, wrapper: F, fn: Callable) -> F:
        wrapper.futures = self._futures  # type: ignore[attr-defined]
        wrapper.afutures = self._afutures  # type: ignore[attr-defined]
        if isinstance(self._cache, _WeakCache):
            wrapper.wrefs = self._cache.alive  # type: ignore[attr-defined]
        if isinstance(self._cache, _StrongCache):
            wrapper.cache = self._cache.cache  # type: ignore[attr-defined]
        functools.update_wrapper(wrapper, fn)
        return wrapper


def call_once[T](fn: Get[T], /) -> Get[T]:
    """Make callable a singleton.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions
    """
    warn(
        'Deprecated. Use `@memoize()` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize()(fn)


def coalesce[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """
    Merge duplicate parallel invocations to the one, keep results til GC.

    Supports async-def functions (but not async-gen functions).
    DO NOT USE with recursive functions
    """
    warn(
        'Deprecated. Use `@memoize(0)` for this',
        DeprecationWarning,
        stacklevel=2,
    )
    return memoize(0)(fn)


_CACHES: dict[tuple[CachePolicy | None, bool], _CacheMaker] = {
    (None, False): _Cache,
    ('lru', False): _LruCache,
    ('mru', False): _MruCache,
    (None, True): _TimedCache,
    ('lru', True): _TimedLruCache,
    ('mru', True): _TimedMruCache,
}
