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
    MutableMapping,
)
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from threading import RLock
from types import CodeType
from typing import Final, Protocol, SupportsInt, cast
from weakref import WeakValueDictionary

from ._dev import declutter_tb
from ._keys import make_key
from ._repr import si_bin
from ._sizeof import sizeof
from ._types import ABatchFn, AnyFuture, BatchFn, CachePolicy, Decorator, KeyFn


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


class _AbstractCache[T](Protocol):
    def __getitem__(self, key: Hashable) -> T | _Empty: ...
    def __setitem__(self, key: Hashable, value: T) -> None: ...


class _CacheMaker[T](Protocol):
    def __call__(
        self, capacity: int, make_node: Callable[[T], _Node[T]]
    ) -> '_AbstractCache[T]': ...


@dataclass(repr=False, slots=True, weakref_slot=True)
class _Cache[T]:
    capacity: int
    make_node: Callable[[T], _Node[T]] = field(repr=False)
    size: int = 0
    store: dict[Hashable, _Node[T]] = field(default_factory=dict)
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


class _Heap[T](_Cache[T]):
    def __getitem__(self, key: Hashable) -> T | _Empty:
        if node := self.store.get(key):
            self.stats.hits += 1
            return node.value

        self.stats.misses += 1
        return _empty

    def __setitem__(self, key: Hashable, value: T) -> None:
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


class _LruMruCache[T](_Cache[T]):
    def __getitem__(self, key: Hashable) -> T | _Empty:
        if node := self.store.pop(key, None):
            self.stats.hits += 1
            self.store[key] = node
            return node.value

        self.stats.misses += 1
        return _empty

    def __setitem__(self, key: Hashable, value: T) -> None:
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


class _LruCache[T](_LruMruCache[T]):
    def pop(self) -> _Node:
        """Drop oldest node."""
        return self.store.pop(next(iter(self.store)))


class _MruCache[T](_LruMruCache[T]):
    def pop(self) -> _Node:
        """Drop most recently added node."""
        return self.store.popitem()[1]


# --------------------------------- utilities --------------------------------


@dataclass(frozen=True, kw_only=True)
class _WeakCache[T]:
    """Retrieve items via weak references from everywhere."""

    alive: WeakValueDictionary[Hashable, T] = field(
        default_factory=WeakValueDictionary
    )

    def __getitem__(self, key: Hashable) -> T | _Empty:
        return self.alive.get(key, _empty)

    def __setitem__(self, key: Hashable, value: T) -> None:
        if type(value).__weakrefoffset__:  # Support weak reference.
            self.alive[key] = value


@dataclass(frozen=True, kw_only=True)
class _StrongCache[R](_WeakCache[R]):
    cache: _AbstractCache[R]

    def __getitem__(self, key: Hashable) -> R | _Empty:
        # Alive and stored items.
        # Called first to update cache stats (i.e. MRU/LRU if any).
        # `cache` has subset of objects from `alive`.
        if (ret := self.cache[key]) is not _empty:
            return ret
        # Item could still exist, try reference ...
        return super().__getitem__(key)

    def __setitem__(self, key: Hashable, value: R) -> None:
        self.cache[key] = value
        super().__setitem__(key, value)


@dataclass(frozen=True, slots=True)
class _CacheState[R]:
    cache: _AbstractCache[R]
    code: CodeType  # for short tracebacks
    key_fn: KeyFn
    futures: WeakValueDictionary[Hashable, AnyFuture[R]] = field(
        default_factory=WeakValueDictionary
    )


# --------------------------------- wrapping ---------------------------------


def _sync_memoize[**P, R](
    fn: Callable[P, R], cs: _CacheState[R]
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
            if not f:
                cs.futures[key] = f = cf.Future[R]()
                is_owner = True

        # Release lock to allow function to run
        if not is_owner:
            return f.result()

        try:
            ret = fn(*args, **kwargs)
        except BaseException as exc:
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


def _async_memoize[**P, R](
    fn: Callable[P, Awaitable[R]],
    cs: _CacheState[R],
) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = cs.key_fn(*args, **kwargs)

        if (ret := cs.cache[key]) is not _empty:
            return ret

        # ... or it could be computed somewhere else, join there.
        if f := cs.futures.get(key):
            assert isinstance(f, asyncio.Future)
            return await f
        cs.futures[key] = f = asyncio.Future[R]()

        # NOTE: fn() is not within threading.Lock, thus it's not thread safe
        # NOTE: but it's async-safe because this `await` is only one here.
        try:
            ret = await fn(*args, **kwargs)
        except BaseException as exc:
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


@dataclass(slots=True, frozen=True)
class _Arg[T]:
    arg: T


class _BatchedQuery[T, R]:
    def __init__(
        self, cs: _CacheState[R], *tokens: T, aio: bool = False
    ) -> None:
        self._cs = cs
        self._keys = [cs.key_fn(t) for t in tokens]  # All keys with duplicates

        self._jobs: list[tuple[Hashable, _Arg[T] | None, AnyFuture[R]]] = []
        self._stash: list[tuple[Hashable, R]] = []
        self._done: dict[Hashable, R] = {}

        for k, t in dict(zip(self._keys, tokens)).items():
            # If this key is processing right now, wait till its done ...
            if f := cs.futures.get(k):  # ! Requires sync
                self._jobs.append((k, None, f))  # Wait for this

            # ... else check if it's done ...
            elif (r := cs.cache[k]) is not _empty:  # ! Requires sync
                self._done[k] = r

            # ... otherwise schedule a new job.
            else:
                f = asyncio.Future[R]() if aio else cf.Future[R]()
                self._jobs.append((k, _Arg(t), f))  # Resolve this manually
                cs.futures[k] = f  # ! Requires sync

        self._errors: dict[BaseException, None] = {}
        self._default_tp: type[BaseException] | None = None

    def __bool__(self) -> bool:
        return bool(self._jobs)

    @property
    def result(self) -> list[R] | BaseException:
        match list(self._errors):
            case []:
                if self._default_tp:
                    return self._default_tp()
                return [self._done[k] for k in self._keys]
            case [e]:
                return e
            case excs:
                msg = 'Got multiple exceptions'
                if all(isinstance(e, Exception) for e in excs):
                    return ExceptionGroup(msg, excs)  # type: ignore[type-var]
                return BaseExceptionGroup(msg, excs)

    @result.setter
    def result(self, obj: list[R] | BaseException) -> None:
        done_jobs = [(k, f) for k, a, f in self._jobs if a]

        if not isinstance(obj, BaseException):
            if len(obj) == len(done_jobs):
                for (k, f), value in zip(done_jobs, obj):
                    f.set_result(value)
                    self._stash.append((k, value))
                return

            obj = RuntimeError(
                f'Call with {len(done_jobs)} arguments '
                f'incorrectly returned {len(obj)} results'
            )

        for _, f in done_jobs:
            f.set_exception(obj)
            if isinstance(f, asyncio.Future):
                f.exception()  # Mark exception as retrieved
        self._errors[obj] = None

    @property
    def args(self) -> list[T]:
        return [a.arg for _, a, _ in self._jobs if a]

    def fs_as[F: AnyFuture](self, tp: type[F]) -> set[F]:
        return {f for _, a, f in self._jobs if not a and isinstance(f, tp)}

    def finalize_fs(self) -> None:
        cerr = cf.CancelledError
        aerr = asyncio.CancelledError
        for k, a, f in self._jobs:
            if a:
                continue  # Our task, not "borrowed" one
            if f.cancelled():
                self._default_tp = cerr if isinstance(f, cf.Future) else aerr
            elif e := f.exception():
                self._errors[e] = None
            else:
                self._stash.append((k, f.result()))

    def sync(self) -> None:
        for e in self._errors:
            declutter_tb(e, self._cs.code)

        for k, r in self._stash:
            self._done[k] = self._cs.cache[k] = r

        # Force next callers to use cache  # ! optional
        for k in self._jobs:
            self._cs.futures.pop(k, None)


def _sync_memoize_batched[T, R](
    fn: BatchFn[T, R], cs: _CacheState[R]
) -> BatchFn[T, R]:
    lock = RLock()

    def wrapper(tokens: Iterable[T]) -> list[R]:
        with lock:
            q = _BatchedQuery(cs, *tokens)

        try:
            # Run tasks we are first to schedule
            if args := q.args:
                try:
                    q.result = list(fn(args))
                except BaseException as exc:  # noqa: BLE001
                    q.result = exc

            # Wait for completion of tasks scheduled by neighbour calls
            if fs := q.fs_as(cf.Future):
                cf.wait(fs)
                q.finalize_fs()
        finally:
            if q:
                with lock:
                    q.sync()

        if isinstance(ret := q.result, BaseException):
            raise ret
        return ret

    return wrapper


def _async_memoize_batched[T, R](
    fn: ABatchFn[T, R], cs: _CacheState[R]
) -> ABatchFn[T, R]:
    async def wrapper(tokens: Iterable[T]) -> list[R]:
        q = _BatchedQuery(cs, *tokens, aio=True)

        try:
            # Run tasks we are first to schedule
            if args := q.args:
                try:
                    q.result = list(await fn(args))
                except BaseException as exc:  # noqa: BLE001
                    q.result = exc  # Raise later in `q.exception()`

            # Wait for completion of tasks scheduled by neighbour calls
            if fs := q.fs_as(asyncio.Future):
                await asyncio.wait(fs)
                q.finalize_fs()
        finally:
            q.sync()

        if isinstance(ret := q.result, BaseException):
            raise ret
        return ret

    return wrapper


# ------------------------------- decorations --------------------------------


def _memoize[**P, R](
    fn: Callable[P, R],
    *,
    cache: _AbstractCache,
    key_fn: KeyFn,
    batched: bool,
) -> Callable[P, R]:
    cs = _CacheState(cache, fn.__code__, key_fn)

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

    while isinstance(cache, _StrongCache):
        cache = cache.cache
    w.cache = cache  # type: ignore[attr-defined]
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
            _memoize, cache=_WeakCache(), batched=batched, key_fn=key_fn
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
            cache=_StrongCache(cache=cache),
            key_fn=key_fn,
            batched=batched,
        )

    msg = f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
    raise ValueError(msg)


_CACHES: dict[CachePolicy, _CacheMaker] = {
    None: _Heap,
    'lru': _LruCache,
    'mru': _MruCache,
}
