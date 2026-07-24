__all__ = ['cache_status', 'memoize']

import asyncio
import concurrent.futures as cf
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
from time import monotonic
from typing import Any, Final, Protocol, SupportsInt, cast
from weakref import WeakValueDictionary

from ._dev import clone_exc, hide_frame
from ._futures import (
    ABatchFn,
    AnyFuture,
    AnyJob,
    BatchFn,
    adispatch,
    dispatch,
    gather_fs,
)
from ._keys import make_key
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import CachePolicy, Decorator, Empty, KeyFn, Some, empty

_inf: Final = float('inf')


@dataclass(repr=False, slots=True)
class _Node[T]:
    value: T
    nbytes: int = 0
    deadline: float | None = None

    def __repr__(self) -> str:
        return repr(self.value)


class _MakeNode:
    def __call__[T](self, obj: T, /) -> _Node[T]:
        return _Node(obj)

    def refresh[T](self, node: _Node[T], now: float, /) -> _Node[T]:
        return node


class _MakeSizedNode(_MakeNode):
    def __call__[T](self, obj: T, /) -> _Node[T]:
        return _Node(obj, sizeof(obj), None)


@dataclass(frozen=True, slots=True)
class _MakeTimedNode(_MakeNode):
    ttl: float

    def __call__[T](self, obj: T, /) -> _Node[T]:
        return _Node(obj, 0, monotonic() + self.ttl)

    def refresh[T](self, node: _Node[T], now: float, /) -> _Node[T]:
        return _Node(node.value, node.nbytes, now + self.ttl)


class _MakeSizedTimedNode(_MakeTimedNode):
    def __call__[T](self, obj: T, /) -> _Node[T]:
        return _Node(obj, sizeof(obj), monotonic() + self.ttl)


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


class _AbstractCache[T](Protocol):
    def __getitem__(self, key: Hashable, /) -> T | Empty: ...
    def __setitem__(self, key: Hashable, value: T, /) -> None: ...


class _CacheMaker[T](Protocol):
    def __call__(
        self, capacity: int, capacity_bytes: int, make_node: _MakeNode
    ) -> _AbstractCache[T]: ...


@dataclass(repr=False, slots=True, weakref_slot=True)
class _Cache[T]:
    capacity: int
    capacity_bytes: int
    make_node: _MakeNode = field(repr=False)
    nbytes: int = 0
    store: dict[Hashable, _Node[T]] = field(default_factory=dict)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self) -> None:
        assert self.capacity != 0
        assert self.capacity_bytes != 0
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

    def _maybe_insert(self, key: Hashable, node: _Node[T], /) -> None:
        if (0 < self.capacity <= len(self.store)) or (
            0 < self.capacity_bytes < self.nbytes + node.nbytes
        ):  # no free space
            return
        self.store[key] = node
        self.nbytes += node.nbytes


class _Heap[T](_Cache[T]):
    """
    No time limit.
    When size limit is reached, nothing is removed
    """

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        if node := self.store.get(key):
            self.stats.hits += 1
            return node.value

        self.stats.misses += 1
        return empty

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        if key not in self.store:
            node = self.make_node(value)
            self._maybe_insert(key, node)


class _TimedCache[T](_Cache[T]):
    """
    Time limit for each entry.
    When size limit is reached, nothing is removed
    """

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        now = monotonic()
        self._remove_outdated(now)

        if node := self.store.pop(key, None):
            self.stats.hits += 1
            self.store[key] = self.make_node.refresh(node, now)  # move front
            return node.value

        self.stats.misses += 1
        return empty

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        now = monotonic()
        node = self.store.pop(key, None)  # pop before GC to reuse size
        self._remove_outdated(now)
        if node:
            self.store[key] = self.make_node.refresh(node, now)  # move front
        else:
            node = self.make_node(value)
            self._maybe_insert(key, node)

    def _remove_outdated(self, now: float) -> None:
        while self.store:
            key, node = next(iter(self.store.items()))
            if (node.deadline or _inf) > now:
                return  # reached alive node before free space
            self.store.pop(key)  # dead node, delete
            self.nbytes -= node.nbytes
            self.stats.dropped += 1


class _TimedRecencyCache[T](_TimedCache[T]):
    """
    Time limit for each entry.
    When size limit is reached, eviction happens.
    """

    def _maybe_insert(self, key: Hashable, node: _Node[T], /) -> None:
        if self.store and len(self.store) == self.capacity:  # no space
            self.nbytes -= self.pop().nbytes  # evict
            self.stats.dropped += 1

        if self.capacity_bytes > 0:  # byte-bound cache
            max_self_bytes_to_fit = self.capacity_bytes - node.nbytes
            if max_self_bytes_to_fit < 0:  # cache will never fit this
                return
            while self.store and self.nbytes > max_self_bytes_to_fit:  # evict
                self.nbytes -= self.pop().nbytes
                self.stats.dropped += 1

        self.store[key] = node
        self.nbytes += node.nbytes

    def pop(self) -> _Node[T]:
        raise NotImplementedError


class _LruTimedCache[T](_TimedRecencyCache[T]):
    """
    Time limit for each entry.
    When size limit is reached, least recently used are evicted
    """

    def pop(self) -> _Node[T]:
        """Drop oldest node."""
        return self.store.pop(next(iter(self.store)))


class _MruTimedCache[T](_TimedRecencyCache[T]):
    """
    Time limit for each entry.
    When size limit is reached, most recently used are evicted
    """

    def pop(self) -> _Node[T]:
        """Drop most recently added node."""
        return self.store.popitem()[1]


# --------------------------------- utilities --------------------------------


@dataclass(frozen=True, kw_only=True)
class _WeakCache[T]:
    """Retrieve items via weak references from everywhere."""

    alive: WeakValueDictionary[Hashable, T] = field(
        default_factory=WeakValueDictionary
    )

    def __getitem__(self, key: Hashable, /) -> T | Empty:
        return self.alive.get(key, empty)

    def __setitem__(self, key: Hashable, value: T, /) -> None:
        if type(value).__weakrefoffset__:  # Support weak reference.
            self.alive[key] = value


@dataclass(frozen=True, kw_only=True)
class _StrongCache[T](_WeakCache[T]):
    cache: _AbstractCache[T]

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


@dataclass(frozen=True, slots=True)
class _CacheState[**P, R]:
    cache: _AbstractCache[R]
    key_fn: KeyFn[P]
    futures: WeakValueDictionary[Hashable, AnyFuture[R]] = field(
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


def _sync_memoize[**P, R](
    fn: Callable[P, R],
    cs: _CacheState[P, R],
) -> Callable[P, R]:
    lock = RLock()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = cs.key_fn(*args, **kwargs)

        is_owner = False
        with lock:
            if (ret := cs.cache[key]) is not empty:
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
            f.set_exception(clone_exc(exc))
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


def _async_memoize[**P, R](
    fn: Callable[P, Awaitable[R]],
    cs: _CacheState[P, R],
) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = cs.key_fn(*args, **kwargs)

        if (ret := cs.cache[key]) is not empty:
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
            f.set_exception(clone_exc(exc))
            cs.futures.pop(key)
            raise
        else:
            f.set_result(ret)
            cs.cache[key] = ret
            cs.futures.pop(key)
            return ret

    return wrapper


# ----------------------- wrapper with batching support ----------------------


class _BatchedQuery[T, R]:
    def __init__(
        self, cs: _CacheState[[T], R], *tokens: T, aio: bool = False
    ) -> None:
        self._cs = cs
        self._keys = [cs.key_fn(t) for t in tokens]  # All keys with duplicates

        self.jobs: list[tuple[Hashable, Some[T] | None, AnyFuture[R]]] = []
        self._done: dict[Hashable, R] = {}

        for k, t in dict(zip(self._keys, tokens)).items():
            # If this key is processing right now, wait till its done ...
            if f := cs.futures.get(k):  # ! Requires sync
                self.jobs.append((k, None, f))  # Wait for this

            # ... else check if it's done ...
            elif (r := cs.cache[k]) is not empty:  # ! Requires sync
                self._done[k] = r

            # ... otherwise schedule a new job.
            else:
                f = (asyncio.Future if aio else cf.Future)[R]()
                self.jobs.append((k, Some(t), f))  # Resolve this manually
                cs.futures[k] = f  # ! Requires sync

    @property
    def pending_jobs(self) -> list[AnyJob[T, R]]:
        return [(a.x, f) for _, a, f in self.jobs if a]

    def running_as[F: AnyFuture](self, tp: type[F]) -> set[F]:
        return {f for _, a, f in self.jobs if not a and isinstance(f, tp)}

    def sync(self, stash: Mapping[Hashable, R]) -> None:
        for k, r in stash.items():
            self._done[k] = self._cs.cache[k] = r

        # Force next callers to use cache  # ! optional
        for k, _, _ in self.jobs:
            self._cs.futures.pop(k, None)

    def result(self) -> list[R]:
        return [self._done[k] for k in self._keys]


def _sync_memoize_batched[T, R](
    fn: BatchFn[T, R], cs: _CacheState[[T], R]
) -> BatchFn[T, R]:
    lock = RLock()

    def wrapper(tokens: Iterable[T]) -> list[R]:
        with lock:
            q = _BatchedQuery(cs, *tokens)

        stash: dict[Hashable, R] = {}
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


def _async_memoize_batched[T, R](
    fn: ABatchFn[T, R], cs: _CacheState[[T], R]
) -> ABatchFn[T, R]:
    async def wrapper(tokens: Iterable[T]) -> list[R]:
        q = _BatchedQuery(cs, *tokens, aio=True)

        stash: dict[Hashable, R] = {}
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


def _memoize[**P, R](
    fn: Callable[P, R],
    *,
    cs: _CacheState[..., Any],
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
    ttl: float | None = None,
) -> Decorator:
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
    if count == 0 and nbytes > 0:
        msg = 'Ambiguous: count=0 disables cache, but nbytes > 0.'
        raise ValueError(msg)
    # +/+, +/0, +/-, 0/0, 0/-, -/+, -/0, -/-
    if count > 0 and nbytes == 0:
        msg = 'Ambiguous: nbytes=0 disables cache, but count > 0.'
        raise ValueError(msg)
    # +/+, +/-, 0/0, 0/-, -/+, -/0, -/-
    if count == 0 or nbytes == 0 or (ttl is not None and ttl <= 0):
        # 0/0, 0/-, -/0 (weakrefs only)
        return functools.partial(  # type: ignore[return-value]
            _memoize,
            cs=_CacheState(_WeakCache(), key_fn),
            batched=batched,
        )

    # +/+(count+nbytes), +/-(count), -/+(nbytes), -/-(unbound)
    if cache_cls := _CACHES.get(policy):
        if (count < 0 and nbytes < 0) or policy is None:  # only time could cap
            cache_cls = _Heap if ttl is None else _TimedCache
        make_node = (
            (_MakeSizedNode() if ttl is None else _MakeSizedTimedNode(ttl))
            if nbytes > 0
            else (_MakeNode() if ttl is None else _MakeTimedNode(ttl))
        )
        cache = cache_cls(count, nbytes, make_node)
        return functools.partial(  # type: ignore[return-value]
            _memoize,
            cs=_CacheState(_StrongCache(cache=cache), key_fn),
            batched=batched,
        )

    msg = f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
    raise ValueError(msg)


_CACHES: dict[CachePolicy, _CacheMaker] = {
    None: _TimedCache,
    'lru': _LruTimedCache,
    'mru': _MruTimedCache,
}
