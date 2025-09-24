__all__ = ['cache_status', 'memoize']

import asyncio
import concurrent.futures as cf
import enum
import functools
from collections.abc import (
    Awaitable,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
)
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from threading import RLock
from typing import Any, Final, Protocol, SupportsInt, cast
from weakref import WeakValueDictionary

from ._dev import clone_exc, hide_frame
from ._futures import adispatch, dispatch, gather_fs
from ._keys import make_key
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import (
    ABatchFn,
    AnyFuture,
    BatchFn,
    CachePolicy,
    Decorator,
    Job,
    KeyFn,
    Some,
)


class _Empty(enum.Enum):
    token = 0


_empty: Final = _Empty.token


@dataclass(repr=False, slots=True)
class _Node[T]:
    value: T
    size: int

    def __repr__(self) -> str:
        return repr(self.value)


def _make_node[T](obj: T, /) -> _Node[T]:
    return _Node(obj, 1)


def _make_sized_node[T](obj: T, /) -> _Node[T]:
    return _Node(obj, sizeof(obj))


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


_REFS: MutableMapping[int, '_Cache'] = WeakValueDictionary()


class _AbstractCache[K: Hashable, T](Protocol):
    def __getitem__(self, key: K, /) -> T | _Empty: ...
    def __setitem__(self, key: K, value: T, /) -> None: ...


class _CacheMaker[K, T](Protocol):
    def __call__(
        self, capacity: int, make_node: Callable[[T], _Node[T]]
    ) -> '_AbstractCache[K, T]': ...


@dataclass(repr=False, slots=True, weakref_slot=True)
class _Cache[K: Hashable, T]:
    capacity: int
    make_node: Callable[[T], _Node[T]] = field(repr=False)
    size: int = 0
    store: dict[K, _Node[T]] = field(default_factory=dict)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self) -> None:
        _REFS[id(self)] = self

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self) -> Iterator:
        return iter(self.store)

    def keys(self) -> KeysView:
        return self.store.keys()

    def clear(self) -> None:
        self.stats.dropped += len(self.store)
        self.store.clear()
        self.size = 0

    def __repr__(self) -> str:
        args = [
            f'items={len(self.store)}',
            f'size={type(self.capacity)(self.size)}',
            f'capacity={self.capacity}',
        ]
        if self.stats:
            args.append(f'stats={self.stats}')
        return f'{type(self).__name__}({", ".join(args)})'


class _Heap[K: Hashable, T](_Cache[K, T]):
    def __getitem__(self, key: K, /) -> T | _Empty:
        if node := self.store.get(key):
            self.stats.hits += 1
            return node.value

        self.stats.misses += 1
        return _empty

    def __setitem__(self, key: K, value: T, /) -> None:
        if key in self.store:
            return
        node = self.make_node(value)
        if (
            self.capacity >= 0  # bound cache
            and self.size + node.size > self.capacity  # no free place
        ):
            return
        self.store[key] = node
        self.size += node.size


class _LruMruCache[K: Hashable, T](_Cache[K, T]):
    def __getitem__(self, key: K, /) -> T | _Empty:
        if node := self.store.pop(key, None):
            self.stats.hits += 1
            self.store[key] = node
            return node.value

        self.stats.misses += 1
        return _empty

    def __setitem__(self, key: K, value: T, /) -> None:
        if key in self.store:
            return
        node = self.make_node(value)
        nsize = node.size
        if self.capacity >= 0:  # bound cache
            if nsize > self.capacity:  # cache will never fit this
                return

            while self.store and self.size + nsize > self.capacity:  # evict
                self.size -= self.pop().size
                self.stats.dropped += 1

        self.store[key] = node
        self.size += nsize

    def pop(self) -> _Node:
        raise NotImplementedError


class _LruCache[K: Hashable, T](_LruMruCache[K, T]):
    def pop(self) -> _Node:
        """Drop oldest node."""
        return self.store.pop(next(iter(self.store)))


class _MruCache[K: Hashable, T](_LruMruCache[K, T]):
    def pop(self) -> _Node:
        """Drop most recently added node."""
        return self.store.popitem()[1]


# --------------------------------- utilities --------------------------------


@dataclass(frozen=True, kw_only=True)
class _WeakCache[K: Hashable, T]:
    """Retrieve items via weak references from everywhere."""

    alive: WeakValueDictionary[K, T] = field(
        default_factory=WeakValueDictionary
    )

    def __getitem__(self, key: K, /) -> T | _Empty:
        return self.alive.get(key, _empty)

    def __setitem__(self, key: K, value: T, /) -> None:
        if type(value).__weakrefoffset__:  # Support weak reference.
            self.alive[key] = value


@dataclass(frozen=True, kw_only=True)
class _StrongCache[K: Hashable, T](_WeakCache[K, T]):
    cache: _AbstractCache[K, T]

    def __getitem__(self, key: K, /) -> T | _Empty:
        # Alive and stored items.
        # Called first to update cache stats (i.e. MRU/LRU if any).
        # `cache` has subset of objects from `alive`.
        if (ret := self.cache[key]) is not _empty:
            return ret
        # Item could still exist, try reference ...
        return super().__getitem__(key)

    def __setitem__(self, key: K, value: T, /) -> None:
        self.cache[key] = value
        super().__setitem__(key, value)


@dataclass(frozen=True, slots=True)
class _CacheState[K: Hashable, R]:
    cache: _AbstractCache[K, R]
    key_fn: KeyFn[K]
    futures: WeakValueDictionary[K, AnyFuture[R]] = field(
        default_factory=WeakValueDictionary
    )


# --------------------------------- wrapping ---------------------------------


def _result[T](f: cf.Future[T]) -> T:
    if f.cancelled():
        with hide_frame:
            raise cf.CancelledError
    if exc := f.exception():
        with hide_frame:
            raise exc
    return f.result()


def _sync_memoize[K: Hashable, **P, R](
    fn: Callable[P, R],
    cs: _CacheState[K, R],
) -> Callable[P, R]:
    lock = RLock()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = cs.key_fn(*args, **kwargs)

        is_owner = False
        with lock:
            if (ret := cs.cache[key]) is not _empty:
                return ret

            # ... or it could be computed somewhere else, join there.
            f = cs.futures.get(key)
            if f:
                assert isinstance(f, cf.Future)
            else:
                cs.futures[key] = f = cf.Future[R]()
                is_owner = True

        # Release lock to allow function to run
        if not is_owner:
            with hide_frame:
                return _result(f)

        try:
            with hide_frame:
                ret = fn(*args, **kwargs)
        except BaseException as exc:
            exc = clone_exc(exc)  # Protect from mutation by outer frame
            f.set_exception(exc)
            with lock:
                cs.futures.pop(key)
            raise
        else:
            f.set_result(ret)
            with lock:
                cs.cache[key] = ret
                cs.futures.pop(key)
            return ret

    return wrapper


def _async_memoize[K: Hashable, **P, R](
    fn: Callable[P, Awaitable[R]],
    cs: _CacheState[K, R],
) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = cs.key_fn(*args, **kwargs)

        if (ret := cs.cache[key]) is not _empty:
            return ret

        # ... or it could be computed somewhere else, join there.
        if f := cs.futures.get(key):
            assert isinstance(f, asyncio.Future)
            with hide_frame:
                return await f
        cs.futures[key] = f = asyncio.Future[R]()

        # NOTE: fn() is not within threading.Lock, thus it's not thread safe
        # NOTE: but it's async-safe because this `await` is only one here.
        try:
            with hide_frame:
                ret = await fn(*args, **kwargs)
        except BaseException as exc:
            exc = clone_exc(exc)
            f.set_exception(exc)
            cs.futures.pop(key)
            raise
        else:
            f.set_result(ret)
            cs.cache[key] = ret
            cs.futures.pop(key)
            return ret

    return wrapper


# ----------------------- wrapper with batching support ----------------------


class _BatchedQuery[K: Hashable, T, R]:
    def __init__(
        self, cs: _CacheState[K, R], *tokens: T, aio: bool = False
    ) -> None:
        self._cs = cs
        self._keys = [cs.key_fn(t) for t in tokens]  # All keys with duplicates

        self.jobs: list[tuple[K, Some[T] | None, AnyFuture[R]]] = []
        self._done: dict[K, R] = {}

        for k, t in dict(zip(self._keys, tokens)).items():
            # If this key is processing right now, wait till its done ...
            if f := cs.futures.get(k):  # ! Requires sync
                self.jobs.append((k, None, f))  # Wait for this

            # ... else check if it's done ...
            elif (r := cs.cache[k]) is not _empty:  # ! Requires sync
                self._done[k] = r

            # ... otherwise schedule a new job.
            else:
                f = asyncio.Future[R]() if aio else cf.Future[R]()
                self.jobs.append((k, Some(t), f))  # Resolve this manually
                cs.futures[k] = f  # ! Requires sync

    @property
    def pending_jobs(self) -> list[Job[T, R]]:
        return [(a.x, f) for _, a, f in self.jobs if a]

    def running_as[F: AnyFuture](self, tp: type[F]) -> set[F]:
        return {f for _, a, f in self.jobs if not a and isinstance(f, tp)}

    def sync(self, stash: Mapping[K, R]) -> None:
        for k, r in stash.items():
            self._done[k] = self._cs.cache[k] = r

        # Force next callers to use cache  # ! optional
        for k, _, _ in self.jobs:
            self._cs.futures.pop(k, None)

    def result(self) -> list[R]:
        return [self._done[k] for k in self._keys]


def _sync_memoize_batched[K: Hashable, T, R](
    fn: BatchFn[T, R], cs: _CacheState[K, R]
) -> BatchFn[T, R]:
    lock = RLock()

    def wrapper(tokens: Iterable[T]) -> list[R]:
        with lock:
            q = _BatchedQuery(cs, *tokens)

        stash: dict[K, R] = {}
        try:
            if jobs := q.pending_jobs:
                dispatch(fn, *jobs)

            if fs := q.running_as(cf.Future):
                cf.wait(fs)

            stash, err = gather_fs((k, f) for k, _, f in q.jobs)
        finally:
            if q.jobs:
                with lock:
                    q.sync(stash)

        if err is None:
            return q.result()
        with hide_frame:
            raise err

    return wrapper


def _async_memoize_batched[K: Hashable, T, R](
    fn: ABatchFn[T, R], cs: _CacheState[K, R]
) -> ABatchFn[T, R]:
    async def wrapper(tokens: Iterable[T]) -> list[R]:
        q = _BatchedQuery(cs, *tokens, aio=True)

        stash: dict[K, R] = {}
        try:
            if jobs := q.pending_jobs:
                await adispatch(fn, *jobs)

            if fs := q.running_as(asyncio.Future):
                await asyncio.wait(fs)

            stash, err = gather_fs((k, f) for k, _, f in q.jobs)
        finally:
            q.sync(stash)

        if err is None:
            return q.result()
        with hide_frame:
            raise err

    return wrapper


# ------------------------------- decorations --------------------------------


def _memoize[K: Hashable, **P, R](
    fn: Callable[P, R],
    *,
    cs: _CacheState[K, Any],
    batched: bool,
) -> Callable[P, R]:
    if batched and iscoroutinefunction(fn):
        w = cast(
            'Callable[P, R]',
            _async_memoize_batched(cast('ABatchFn', fn), cs=cs),
        )
    elif batched:
        w = cast(
            'Callable[P, R]',
            _sync_memoize_batched(cast('BatchFn', fn), cs=cs),
        )
    elif iscoroutinefunction(fn):
        w = cast('Callable[P, R]', _async_memoize(fn, cs=cs))
    else:
        w = _sync_memoize(fn, cs=cs)

    w.running = cs.futures  # type: ignore[attr-defined]
    if isinstance(cs.cache, _WeakCache):
        w.wrefs = cs.cache.alive  # type: ignore[attr-defined]
    if isinstance(cs.cache, _StrongCache):
        w.cache = cs.cache.cache  # type: ignore[attr-defined]

    return functools.update_wrapper(w, fn)


# ----------------------------- factory wrappers -----------------------------


def memoize(
    count: SupportsInt | None = None,
    *,
    nbytes: SupportsInt | None = None,
    batched: bool = False,
    policy: CachePolicy = None,
    key_fn: KeyFn = make_key,
) -> Decorator:
    """Create caching decorator.

    Parameters:
    - count - max objects to store or None for unbound cache.
    - nbytes - max bytes to store.
    - policy - eviction policy, either "raw" (no eviction), or "lru"
      (evict oldest), or "mru" (evict most recent).
    - batched - set if callable supports batching.
    """
    count = -1 if count is None else int(count)
    nbytes = -1 if nbytes is None else si_bin(int(nbytes))
    if count >= 0 and nbytes >= 0:
        msg = 'Only one of `count`/`nbytes` can be used. Not both'
        raise ValueError(msg)

    # count/nbytes in -/- (unbound), -/0 or 0/- (off), -/+ (bytes), +/- (count)
    capacity = max(count, nbytes)
    if int(capacity) == 0:
        return functools.partial(  # type: ignore[return-value]
            _memoize,
            cs=_CacheState(_WeakCache(), key_fn),
            batched=batched,
        )

    if cache_cls := _CACHES.get(policy):
        make_node = _make_node
        # count/nbytes in -/- (unbound), -/+ (bytes), +/- (count)
        if capacity < 0:
            cache_cls = _Heap
        # count/nbytes in -/+ (bytes), +/- (count)
        elif nbytes > 0:
            make_node = _make_sized_node

        cache = cache_cls(capacity, make_node)
        return functools.partial(  # type: ignore[return-value]
            _memoize,
            cs=_CacheState(_StrongCache(cache=cache), key_fn),
            batched=batched,
        )

    msg = f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
    raise ValueError(msg)


_CACHES: dict[CachePolicy, _CacheMaker] = {
    None: _Heap,
    'lru': _LruCache,
    'mru': _MruCache,
}
