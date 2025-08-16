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
from contextlib import ExitStack
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from threading import RLock
from typing import Final, Literal, NamedTuple, Protocol, SupportsInt, cast
from weakref import WeakValueDictionary

from ._keys import make_key
from ._repr import si_bin
from ._sizeof import sizeof


class _Empty(enum.Enum):
    token = 0


type _BatchedFn[T, R] = Callable[[list[T]], Iterable[R]]
type _AsyncBatchedFn[T, R] = Callable[[list[T]], Awaitable[Iterable[R]]]
type _Policy = Literal['lru', 'mru'] | None
type _KeyFn = Callable[..., Hashable]

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
    def __init__(
        self, capacity: int, make_node: Callable[[T], _Node[T]]
    ) -> None: ...
    def __getitem__(self, key: Hashable) -> T | _Empty: ...
    def __setitem__(self, key: Hashable, value: T) -> None: ...


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


# -------------------------------- wrapping --------------------------------


@dataclass
class _Memoize[**P, R, R1]:
    fn: Callable[P, R]
    key_fn: _KeyFn
    cache: _AbstractCache[R1] | None
    alive: WeakValueDictionary[Hashable, R1] = field(
        default_factory=WeakValueDictionary
    )


@dataclass
class _SyncMemoize[**P, R](_Memoize[P, R, R]):
    running: dict[Hashable, cf.Future[R]] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = self.key_fn(*args, **kwargs)

        is_owner = False
        with self.lock:
            # Alive and stored items.
            # Called first to update cache stats (i.e. MRU/LRU if any).
            # `cache` has subset of objects from `alive`.
            if (
                self.cache is not None
                and (ret := self.cache[key]) is not _empty
            ):
                return ret

            # Item could still exist, try reference ...
            if (ret := self.alive.get(key, _empty)) is not _empty:
                return ret
            # ... or it could be computed somewhere else, join there.
            f = self.running.get(key)
            if not f:
                self.running[key] = f = cf.Future()
                is_owner = True

        # Release lock to allow function to run
        if not is_owner:
            return f.result()
        try:
            ret = self.fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(exc)
            with self.lock:
                self.running.pop(key)
            raise
        else:
            f.set_result(ret)
            with self.lock:
                if self.cache is not None:
                    self.cache[key] = ret
                if type(ret).__weakrefoffset__:  # Support weak reference.
                    self.alive[key] = ret
                self.running.pop(key)
            return ret


@dataclass
class _AsyncMemoize[**P, R](_Memoize[P, Awaitable[R], R]):
    running: dict[Hashable, asyncio.Future[R]] = field(default_factory=dict)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = self.key_fn(*args, **kwargs)

        # Alive and stored items.
        # Called first to update cache stats (i.e. MRU/LRU if any).
        # `cache` has subset of objects from `alive`.
        if self.cache is not None and (ret := self.cache[key]) is not _empty:
            return ret

        # Item could still exist, try reference ...
        if (ret := self.alive.get(key, _empty)) is not _empty:
            return ret

        # ... or it could be computed somewhere else, join there.
        if f := self.running.get(key):
            return await f
        self.running[key] = f = asyncio.get_running_loop().create_future()

        # NOTE: fn() is not within threading.Lock, thus it's not thread safe
        # NOTE: but it's async-safe because this `await` is only one here.
        try:
            ret = await self.fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(exc)
            self.running.pop(key)
            raise
        else:
            f.set_result(ret)
            if self.cache is not None:
                self.cache[key] = ret
            if type(ret).__weakrefoffset__:  # Support weak reference.
                self.alive[key] = ret
            self.running.pop(key)
            return ret


# ----------------------- wrapper with batching support ----------------------


class _Job[T, R](NamedTuple):
    token: T
    future: cf.Future[R]


class _AsyncJob[T, R](NamedTuple):
    token: T
    future: asyncio.Future[R]


@dataclass(kw_only=True)
class _MemoizeBatched[T, R]:
    key_fn: _KeyFn
    cache: _AbstractCache[R] | None
    alive: WeakValueDictionary[Hashable, R] = field(
        default_factory=WeakValueDictionary
    )

    def _merge_inputs(self, tokens: Iterable[T]) -> tuple[
        list[tuple[Hashable, T]],
        dict[Hashable, R],
        dict[Hashable, T],
    ]:
        keyed_tokens = [(self.key_fn(t), t) for t in tokens]

        hits: dict[Hashable, R] = {}
        misses: dict[Hashable, T] = {}
        for k, t in dict(keyed_tokens).items():
            if self.cache is not None and (r := self.cache[k]) is not _empty:
                hits[k] = r
            else:
                try:
                    r = self.alive[k]
                except KeyError:
                    misses[k] = t
                else:
                    hits[k] = r
        return keyed_tokens, hits, misses

    def _get_results(
        self,
        keyed_tokens: list[tuple[Hashable, T]],
        hits: dict[Hashable, R],
        it: Iterable[tuple[Hashable, R]],
    ) -> list[R]:
        for k, r in it:
            hits[k] = r
            if self.cache is not None:
                self.cache[k] = r
            if type(r).__weakrefoffset__:  # Support weak reference.
                self.alive[k] = r
        return [hits[k] for k, _ in keyed_tokens]


@dataclass
class _SyncMemoizeBatched[T, R](_MemoizeBatched[T, R]):
    fn: _BatchedFn[T, R]
    lock: RLock = field(default_factory=RLock)
    running: WeakValueDictionary[Hashable, cf.Future[R]] = field(
        default_factory=WeakValueDictionary
    )
    queue: dict[Hashable, _Job[T, R]] = field(default_factory=dict)

    def _dispatch(self) -> None:
        jobs = {**self.queue}
        self.queue.clear()

        try:
            values = [*self.fn([job.token for job in jobs.values()])]

            if len(values) != len(jobs):
                msg = (
                    'Input batch size is not equal to output: '
                    f'{len(values)} != {len(jobs)}'
                )
                raise RuntimeError(msg)  # noqa: TRY301

        except BaseException as exc:  # noqa: BLE001
            for key, job in jobs.items():
                self.running.pop(key)
                job.future.set_exception(exc)

        else:
            for job, value in zip(jobs.values(), values):
                job.future.set_result(value)

    def _load(self, stack: ExitStack, key: Hashable, token: T) -> cf.Future[R]:
        with self.lock:
            if result := self.running.get(key):
                return result

            self.running[key] = future = cf.Future[R]()
            self.queue[key] = _Job(token, future)
            if len(self.queue) == 1:
                stack.callback(self._dispatch)

        return future

    def __call__(self, tokens: Iterable[T]) -> list[R]:
        with self.lock:
            keyed_tokens, hits, misses = self._merge_inputs(tokens)

        futs: dict[Hashable, cf.Future] = {}
        with ExitStack() as stack:
            futs |= {k: self._load(stack, k, t) for k, t in misses.items()}
        cf.wait(futs.values(), return_when='FIRST_EXCEPTION')

        # Process misses
        it = ((k, f.result()) for k, f in futs.items())
        with self.lock:
            return self._get_results(keyed_tokens, hits, it)


@dataclass
class _AsyncMemoizeBatched[T, R](_MemoizeBatched[T, R]):
    fn: _AsyncBatchedFn[T, R]
    running: WeakValueDictionary[Hashable, asyncio.Future[R]] = field(
        default_factory=WeakValueDictionary
    )
    queue: dict[Hashable, _AsyncJob[T, R]] = field(default_factory=dict)

    async def _dispatch(self) -> None:
        jobs = {**self.queue}
        self.queue.clear()

        try:
            values = [*await self.fn([job.token for job in jobs.values()])]

            if len(values) != len(jobs):
                msg = (
                    'Input batch size is not equal to output: '
                    f'{len(values)} != {len(jobs)}'
                )
                raise RuntimeError(msg)  # noqa: TRY301

        except BaseException as exc:  # noqa: BLE001
            for key, job in jobs.items():
                self.running.pop(key)
                job.future.set_exception(exc)

        else:
            for job, value in zip(jobs.values(), values):
                job.future.set_result(value)

    async def _load(self, token: T) -> R:
        key = self.key_fn(token)
        if result := self.running.get(key):
            return await result

        loop = asyncio.get_running_loop()
        self.running[key] = future = loop.create_future()
        self.queue[key] = _AsyncJob(token, future)
        if len(self.queue) == 1:
            await self._dispatch()

        return await future

    async def __call__(self, tokens: Iterable[T]) -> list[R]:
        keyed_tokens, hits, misses = self._merge_inputs(tokens)

        rs = await asyncio.gather(*(self._load(t) for t in misses.values()))

        # Process misses
        it = zip(misses, rs)
        return self._get_results(keyed_tokens, hits, it)


# ------------------------------- decorations --------------------------------


def _memoize[**P, R](
    fn: Callable[P, R],
    *,
    cache: _AbstractCache | None = None,
    key_fn: _KeyFn = make_key,
    batched: bool,
) -> Callable[P, R]:
    if batched and iscoroutinefunction(fn):
        wrapper = cast(
            'Callable[P, R]',
            _AsyncMemoizeBatched(
                cast('_AsyncBatchedFn', fn), key_fn=key_fn, cache=cache
            ),
        )
    elif batched:
        wrapper = cast(
            'Callable[P, R]',
            _SyncMemoizeBatched(
                cast('_BatchedFn', fn), key_fn=key_fn, cache=cache
            ),
        )
    elif iscoroutinefunction(fn):
        wrapper = cast('Callable[P, R]', _AsyncMemoize(fn, key_fn, cache))
    else:
        wrapper = cast('Callable[P, R]', _SyncMemoize(fn, key_fn, cache))

    return functools.update_wrapper(wrapper, fn)


# ----------------------------- factory wrappers -----------------------------


def memoize[**P, T, R](
    count: SupportsInt | None = None,
    *,
    nbytes: SupportsInt | None = None,
    batched: bool = False,
    policy: _Policy = None,
    key_fn: _KeyFn = make_key,
) -> (
    Callable[[Callable[P, R]], Callable[P, R]]
    | Callable[[_BatchedFn[T, R]], _BatchedFn[T, R]]
):
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
            _memoize, key_fn=key_fn, batched=batched
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
            _memoize, cache=cache, key_fn=key_fn, batched=batched
        )

    msg = f'Unknown cache policy: "{policy}". Available: "{set(_CACHES)}"'
    raise ValueError(msg)


_CACHES: dict[_Policy, type[_AbstractCache]] = {
    None: _Heap,
    'lru': _LruCache,
    'mru': _MruCache,
}
