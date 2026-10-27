__all__ = [
    'cache_status',
    'call_once',
    'coalesce',
    'memoize',
    'new_cache',
]

import asyncio
import concurrent.futures as cf
import functools
import inspect
from collections.abc import Callable, Hashable, Iterable, Iterator
from threading import RLock
from time import monotonic
from typing import Final, SupportsInt
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
    dispatch,
    fs_to_results,
    seqcheck,
)
from ._keys import make_key
from ._locking import await_future
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import (
    ACallable,
    Cache,
    CachePolicy,
    Empty,
    Get,
    KeyFn,
    Maybe,
    empty,
)

_inf: Final = float('inf')


class _Node[T]:
    __slots__ = ('nbytes', 'value')

    def __init__(self, value: T, nbytes: int = 0) -> None:
        self.value = value
        self.nbytes = nbytes

    def __repr__(self) -> str:
        line = f'{type(self).__name__}({self.value!r}'
        if self.nbytes > 0:
            line += f' of {si_bin(self.nbytes)}'
        line += ')'
        return line


class Stats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.dropped = 0

    def __bool__(self) -> bool:
        return any(self.__dict__.values())

    def __repr__(self) -> str:
        fields = ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if v)
        return f'{self.__class__.__name__}({fields})'


# ----------------------------- basic caches ------------------------------


def new_cache(
    count: SupportsInt | None = None,
    *,
    nbytes: SupportsInt | None = None,
    policy: CachePolicy | None = None,
    ttl: float = _inf,
    renew_on_access: bool = True,
    weak_fallback: bool = True,
) -> Cache:
    count = -1 if count is None else int(count)
    nbytes = -1 if nbytes is None else int(nbytes)
    if ttl <= 0:
        ttl = _inf

    # +/+, +/0, +/-, 0/+, 0/0, 0/-, -/+, -/0, -/-
    if (count == 0 and nbytes > 0) or (count > 0 and nbytes == 0):
        raise ValueError(
            'Ambiguity: if one of count/nbytes is 0,'
            f'then other should be 0 or -1. Got: {count} and {nbytes}'
        )
    if count < 0 and nbytes < 0:  # Unbound cache, eviction policy is useless
        policy = None

    # +/+, +/-, 0/0, 0/-, -/+, -/0, -/-
    if count == 0 or nbytes == 0:
        # 0/0, 0/-, -/0 (weakrefs only)
        count = nbytes = 0  # Normalize
        if ttl >= _inf:
            return _WeakCache() if weak_fallback else _ZeroCache()
        return _TimedCache(0, 0, ttl, renew_on_access, weak_fallback)

    # +/+(count+nbytes), +/-(count), -/+(nbytes), -/-(unbound)
    if cache_cls := _CACHES.get((policy, ttl < _inf)):
        return cache_cls(count, nbytes, ttl, renew_on_access, weak_fallback)
    raise ValueError(
        f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
    )


def cache_status() -> str:
    return '\n'.join(
        f'{id_:x}: {value!r}' for id_, value in sorted(_REFS.items())
    )


_REFS = WeakValueDictionary[int, '_BaseCache']()


class _ZeroCache:
    def __len__(self) -> int:
        return 0

    def keys(self) -> Iterator[Hashable]:
        return iter(())

    __iter__ = keys

    def clear(self) -> None:
        pass

    def __repr__(self) -> str:
        return f'{type(self).__name__}(items=0)'

    def get(self, key: Hashable, /) -> Empty:
        return empty

    def pop(self, key: Hashable, /) -> Empty:
        return empty

    def __getitem__(self, key: Hashable, /) -> Empty:
        return empty

    def __setitem__(self, key: Hashable, value: object, /) -> None:
        pass

    def __delitem__(self, key: Hashable, /) -> None:
        pass


class _BaseCache[K, T]:
    __slots__ = (
        '__weakref__',
        'capacity',
        'capacity_bytes',
        'deadlines',
        'nbytes',
        'renew_on_access',
        'stats',
        'store',
        'ttl',
        'weak_fallback',
        'wstore',
    )

    def __init__(
        self,
        capacity: int,
        capacity_bytes: int,
        ttl: float = _inf,
        renew_on_access: bool = True,
        weak_fallback: bool = True,
        *,
        store: dict[K, _Node[T]] | None = None,
    ) -> None:
        self.capacity = capacity
        self.capacity_bytes = capacity_bytes
        self.ttl = ttl
        self.renew_on_access = renew_on_access if ttl < _inf else False
        self.weak_fallback = weak_fallback
        self.store = {} if store is None else store
        self.wstore = WeakValueDictionary[K, T]()
        self.deadlines: dict[K, float] = {}
        self.nbytes = 0
        self.stats = Stats()
        _REFS[id(self)] = self

    def __len__(self) -> int:
        self._prune_and_get_new_deadline()
        return len(self.store) + len(self.wstore)

    def keys(self) -> Iterator[K]:
        self._prune_and_get_new_deadline()
        return iter([*self.store, *self.wstore])

    __iter__ = keys

    def clear(self) -> None:
        self.stats.dropped += len(self.store)
        self.deadlines.clear()
        self.store.clear()
        self.wstore.clear()
        self.nbytes = 0

    def __repr__(self) -> str:
        self._prune_and_get_new_deadline()
        args = [
            f'items={len(self.store)}',
            f'weakitems={len(self.wstore)}',
            f'size[bytes]={si_bin(self.nbytes)}',
            f'capacity={self.capacity}',
            f'capacity[bytes]={si_bin(self.capacity_bytes)}',
        ]
        if self.stats:
            args.append(f'stats={self.stats}')
        return f'{type(self).__name__}({", ".join(args)})'

    def get(self, key: K, /) -> T | Empty:
        """Retrieve value by key, access is not recorded"""
        self._prune_and_get_new_deadline()

        if node := self.store.get(key):
            self.stats.hits += 1
            return node.value

        if (value := self.wstore.get(key, empty)) is not empty:
            self.stats.hits += 1
            return value

        self.stats.misses += 1
        return empty

    def pop(self, key: K, /) -> T | Empty:
        """Retrieve value and delete it from cache"""
        self._prune_and_get_new_deadline()
        self.deadlines.pop(key, None)
        if node := self.store.pop(key, None):
            self.nbytes -= node.nbytes
            return node.value
        return self.wstore.pop(key, empty)

    def __getitem__(self, key: K, /) -> T | Empty:
        """Retrieve value and update its atime"""
        deadline = self._prune_and_get_new_deadline()

        if node := self.store.pop(key, None):
            self.stats.hits += 1
            self.store[key] = node  # Move front
            if self.renew_on_access:
                del self.deadlines[key]
                self.deadlines[key] = deadline  # Move front
            return node.value

        if (value := self.wstore.pop(key, empty)) is not empty:
            self.stats.hits += 1
            if self.renew_on_access:
                self.deadlines.pop(key, None)
                self._maybe_insert(key, value, deadline)
            else:
                self._maybe_insert(key, value, _inf)
            return value

        self.stats.misses += 1
        return empty

    def __setitem__(self, key: K, value: T, /) -> None:
        self.getset(key, value)

    def getset(self, key: K, value: T, /) -> T | Empty:
        """Sets the value at key to value and returns the old value at key."""
        deadline = self._prune_and_get_new_deadline()
        self.deadlines.pop(key, None)

        if node := self.store.pop(key, None):
            # Cache hit - update access (assuming value's size is the same)
            if node.value is value:
                self.store[key] = node  # Move front
                if deadline < _inf:
                    self.deadlines[key] = deadline
                return value

            # Cache hit - overwrite (value's size maybe has changed)
            self.nbytes -= node.nbytes
            self._maybe_insert(key, value, deadline)
            return node.value

        old_value = self.wstore.pop(key, empty)
        if old_value is not empty:  # Weak cache hit - update
            self.stats.hits += 1

        self._maybe_insert(key, value, deadline)
        return old_value

    def __delitem__(self, key: K, /) -> None:
        self.deadlines.pop(key, None)
        if node := self.store.pop(key, None):
            self.nbytes -= node.nbytes
            self.stats.dropped += 1
        else:
            self.wstore.pop(key, None)
        self._prune_and_get_new_deadline()

    def _maybe_insert(self, key: K, value: T, deadline: float) -> None:
        if 0 <= self.capacity <= len(self.store):  # No free slots
            if (
                self.weak_fallback
                and _weak_insert(self.wstore, key, value)
                and deadline < _inf
            ):
                self.deadlines[key] = deadline
            return

        nbytes = 0
        do_write = True
        if self.capacity_bytes > 0:  # Has size limit
            nbytes = int(sizeof(value))
            do_write = self.nbytes + nbytes <= self.capacity_bytes  # Node fits

        if do_write:
            self.store[key] = _Node(value, nbytes)
            self.nbytes += nbytes
            if deadline < _inf:
                self.deadlines[key] = deadline
        elif (
            self.weak_fallback
            and _weak_insert(self.wstore, key, value)
            and deadline < _inf
        ):
            self.deadlines[key] = deadline

    def _prune_and_get_new_deadline(self) -> float:
        return _inf


class _TimedCache[K, V](_BaseCache[K, V]):
    """
    Drops items older than `now - ttl`
    """

    def _prune_and_get_new_deadline(self) -> float:
        if self.ttl >= _inf:
            return _inf
        now = monotonic()
        deadline = now + self.ttl
        while self.deadlines:
            key, expires_at = next(iter(self.deadlines.items()))
            if key not in self.store and key not in self.wstore:  # Do GC
                del self.deadlines[key]
                continue
            if expires_at > now:  # Reached alive node before free space
                return deadline
            # Dead node, delete
            if node := self.store.pop(key, None):
                self.nbytes -= node.nbytes  # In main store
            else:
                self.wstore.pop(key, None)  # Maybe in weakref store
            del self.deadlines[key]
            self.stats.dropped += 1
        return deadline


class _EvictableCache[K, V](_BaseCache[K, V]):
    """
    Evicts nodes when cache is too large
    """

    def _maybe_insert(self, key: K, value: V, deadline: float) -> None:
        if self.store and len(self.store) == self.capacity:  # no space
            self._evict()

        if self.capacity_bytes > 0:  # byte-bound cache
            nbytes = int(sizeof(value))
            if nbytes > self.capacity_bytes:  # cache will never fit this
                if (
                    self.weak_fallback
                    and _weak_insert(self.wstore, key, value)
                    and deadline < _inf
                ):
                    self.deadlines[key] = deadline
                return
            while self.store and self.nbytes + nbytes > self.capacity_bytes:
                self._evict()
        else:
            nbytes = 0

        self.store[key] = _Node(value, nbytes)
        if deadline < _inf:
            self.deadlines[key] = deadline
        self.nbytes += nbytes

    def _evict(self) -> None:
        raise NotImplementedError


class _LruCache[K, V](_EvictableCache[K, V]):
    """Evicts least recently used node when cache is too large."""

    def _evict(self) -> None:
        """Drop oldest node."""
        key = next(iter(self.store))
        node = self.store.pop(key)
        self.nbytes -= node.nbytes
        if not (
            self.weak_fallback and _weak_insert(self.wstore, key, node.value)
        ):
            self.deadlines.pop(key, None)
        self.stats.dropped += 1


class _MruCache[K, V](_EvictableCache[K, V]):
    """Evicts most recently used node when cache is too large."""

    def _evict(self) -> None:
        """Drop most recently added node."""
        key, node = self.store.popitem()
        self.nbytes -= node.nbytes
        if not (
            self.weak_fallback and _weak_insert(self.wstore, key, node.value)
        ):
            self.deadlines.pop(key, None)
        self.stats.dropped += 1


class _TimedLruCache[K, V](_LruCache, _TimedCache[K, V]):
    pass


class _TimedMruCache[K, V](_MruCache, _TimedCache[K, V]):
    pass


# --------------------------------- utilities --------------------------------


class _WeakCache[K, V]:
    """Retrieve items via weak references from everywhere."""

    def __init__(self) -> None:
        self.alive = WeakValueDictionary[K, V]()

    def keys(self) -> Iterator[Hashable]:
        return iter(self.alive)

    __iter__ = keys

    def get(self, key: K, /) -> V | Empty:
        return self.alive.get(key, empty)

    def __getitem__(self, key: K, /) -> V | Empty:
        return self.alive.get(key, empty)

    def __setitem__(self, key: K, value: V, /) -> None:
        _weak_insert(self.alive, key, value)

    def __delitem__(self, key: K, /) -> None:
        self.alive.pop(key, None)


def _weak_insert[K, V](s: WeakValueDictionary[K, V], key: K, value: V) -> bool:
    if type(value).__weakrefoffset__:  # Supports weak reference.
        s[key] = value  # Overwrite
        return True

    s.pop(key, None)  # Drop old value
    return False


# ----------------------- wrapper with batching support ----------------------


def _prepare_batch[K, T, F: AnyFuture, V](
    cache: Cache[K, V],
    futures: WeakValueDictionary[K, F],
    new_future: type[F],
    *keyed_tokens: tuple[K, T],
) -> tuple[
    list[K],  # keys
    list[tuple[T, F]],  # pending
    set[F],  # running
    dict[K, F],  # futures
    dict[K, V],  # done
]:
    done: dict[K, V] = {}
    running = set[F]()  # Wait for these
    pending: list[tuple[T, F]] = []  # Populate those
    fs: dict[K, F] = {}

    for k, t in dict(keyed_tokens).items():
        # If this key is processing right now, wait till its done ...
        if (f := futures.get(k)) and not f.cancelled():  # ! Protect
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
    - ttl - time for object to live in the cache. Not renewed by `get()`
    - renew_on_access - if set, `[]` access will renew TTL.
    - weak_fallback - keep values that do not fit in the strong cache
      through weak references; TTL still applies.

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
        ttl: float = _inf,
        renew_on_access: bool = True,
        weak_fallback: bool = True,
    ) -> None:
        self._cache = new_cache(
            count,
            nbytes=nbytes,
            policy=policy,
            ttl=ttl,
            renew_on_access=renew_on_access,
            weak_fallback=weak_fallback,
        )
        self._key_fn = key_fn
        self._batched = batched
        self._lock = RLock()
        self._futures = WeakValueDictionary[Hashable, cf.Future]()
        self._afutures = WeakValueDictionary[Hashable, asyncio.Future]()
        self._is_async: bool | None = None

    def __call__(self, fn: Callable) -> Callable:
        w = (
            (self._awrap_batched(fn) if self._batched else self._awrap(fn))
            if self._do_async(fn)
            else (self._wrap_batched(fn) if self._batched else self._wrap(fn))
        )
        return self._update_wrapper(w, fn)

    def drop(self, fn: Callable) -> Callable:
        w = (
            (self._adrop_batched(fn) if self._batched else self._adrop(fn))
            if self._do_async(fn)
            else (self._drop_batched(fn) if self._batched else self._drop(fn))
        )
        return self._update_wrapper(w, fn)

    def _do_async(self, fn: Callable) -> bool:
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
        return self._is_async

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
                    obj = await_future(f)
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
                    return await asyncio.shield(f)
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
                        await_future(f)
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
                    obj: Maybe[R] = seqcheck(
                        fn(list(ukts.values())), len(ukts)
                    )
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
                    obj: Maybe[R] = seqcheck(
                        await fn(list(ukts.values())), len(ukts)
                    )
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
        else:
            wrapper.cache = self._cache  # type: ignore[attr-defined]
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


_CACHES: dict[
    tuple[CachePolicy | None, bool],
    Callable[[int, int, float, bool, bool], Cache],
] = {
    (None, False): _BaseCache,
    ('lru', False): _LruCache,
    ('mru', False): _MruCache,
    (None, True): _TimedCache,
    ('lru', True): _TimedLruCache,
    ('mru', True): _TimedMruCache,
}
