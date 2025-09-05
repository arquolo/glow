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


def _sync_memoize[**P, R](
    fn: Callable[P, R],
    key_fn: _KeyFn,
    cache: _AbstractCache[R] | None,
) -> Callable[P, R]:
    alive = WeakValueDictionary[Hashable, R]()
    running: dict[Hashable, cf.Future[R]] = {}
    lock = RLock()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = key_fn(*args, **kwargs)

        is_owner = False
        with lock:
            # Alive and stored items.
            # Called first to update cache stats (i.e. MRU/LRU if any).
            # `cache` has subset of objects from `alive`.
            if cache is not None and (ret := cache[key]) is not _empty:
                return ret

            # Item could still exist, try reference ...
            if (ret := alive.get(key, _empty)) is not _empty:
                return ret
            # ... or it could be computed somewhere else, join there.
            f = running.get(key)
            if not f:
                running[key] = f = cf.Future()
                is_owner = True

        # Release lock to allow function to run
        if not is_owner:
            return f.result()
        try:
            ret = fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(exc)
            with lock:
                running.pop(key)
            raise
        else:
            f.set_result(ret)
            with lock:
                if cache is not None:
                    cache[key] = ret
                if type(ret).__weakrefoffset__:  # Support weak reference.
                    alive[key] = ret
                running.pop(key)
            return ret

    return functools.update_wrapper(wrapper, fn)


def _async_memoize[**P, R](
    fn: Callable[P, Awaitable[R]],
    key_fn: _KeyFn,
    cache: _AbstractCache[R] | None,
) -> Callable[P, Awaitable[R]]:
    alive = WeakValueDictionary[Hashable, R]()
    running: dict[Hashable, asyncio.Future[R]] = {}

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = key_fn(*args, **kwargs)

        # Alive and stored items.
        # Called first to update cache stats (i.e. MRU/LRU if any).
        # `cache` has subset of objects from `alive`.
        if cache is not None and (ret := cache[key]) is not _empty:
            return ret

        # Item could still exist, try reference ...
        if (ret := alive.get(key, _empty)) is not _empty:
            return ret

        # ... or it could be computed somewhere else, join there.
        if f := running.get(key):
            return await f
        running[key] = f = asyncio.Future()

        # NOTE: fn() is not within threading.Lock, thus it's not thread safe
        # NOTE: but it's async-safe because this `await` is only one here.
        try:
            ret = await fn(*args, **kwargs)
        except BaseException as exc:
            f.set_exception(exc)
            running.pop(key)
            raise
        else:
            f.set_result(ret)
            if cache is not None:
                cache[key] = ret
            if type(ret).__weakrefoffset__:  # Support weak reference.
                alive[key] = ret
            running.pop(key)
            return ret

    return functools.update_wrapper(wrapper, fn)


# ----------------------- wrapper with batching support ----------------------


class _Job[T, R](NamedTuple):
    token: T
    future: cf.Future[R]


class _AsyncJob[T, R](NamedTuple):
    token: T
    future: asyncio.Future[R]


@dataclass(frozen=True)
class _Adapter[T, R]:
    key_fn: _KeyFn
    cache: _AbstractCache[R] | None
    alive: WeakValueDictionary[Hashable, R] = field(
        default_factory=WeakValueDictionary
    )

    def merge_inputs(self, tokens: Iterable[T]) -> tuple[
        tuple[
            list[tuple[Hashable, T]],
            dict[Hashable, R],
        ],
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
        return (keyed_tokens, hits), misses

    def get_results(
        self,
        keyed_tokens_n_hits: tuple[
            list[tuple[Hashable, T]],
            dict[Hashable, R],
        ],
        done: dict[Hashable, R],
    ) -> list[R]:
        keyed_tokens, hits = keyed_tokens_n_hits
        for k, r in done.items():
            hits[k] = r
            if self.cache is not None:
                self.cache[k] = r
            if type(r).__weakrefoffset__:  # Support weak reference.
                self.alive[k] = r
        return [hits[k] for k, _ in keyed_tokens]


def _sync_memoize_batched[T, R](  # noqa: C901
    fn: _BatchedFn[T, R], adapter: _Adapter[T, R]
) -> _BatchedFn[T, R]:
    lock = RLock()
    running = WeakValueDictionary[Hashable, cf.Future[R]]()
    queue: dict[Hashable, _Job[T, R]] = {}

    def _dispatch() -> None:
        jobs = {**queue}
        queue.clear()

        try:
            values = [*fn([job.token for job in jobs.values()])]

            if len(values) != len(jobs):
                msg = (
                    'Input batch size is not equal to output: '
                    f'{len(values)} != {len(jobs)}'
                )
                raise RuntimeError(msg)  # noqa: TRY301

        except BaseException as exc:  # noqa: BLE001
            for key, job in jobs.items():
                running.pop(key)
                job.future.set_exception(exc)

        else:
            for job, value in zip(jobs.values(), values):
                job.future.set_result(value)

    def _load(stack: ExitStack, key: Hashable, token: T) -> cf.Future[R]:
        if f := running.get(key):
            return f

        running[key] = f = cf.Future[R]()
        queue[key] = _Job(token, f)
        if len(queue) == 1:
            stack.callback(_dispatch)

        return f

    def wrapper(tokens: Iterable[T]) -> list[R]:
        futs: dict[Hashable, cf.Future] = {}

        with ExitStack() as stack, lock:
            keyed_tokens_hits, misses = adapter.merge_inputs(tokens)
            futs |= {k: _load(stack, k, t) for k, t in misses.items()}

        cf.wait(futs.values(), return_when='FIRST_EXCEPTION')

        # Process misses
        done = {k: f.result() for k, f in zip(misses, futs.values())}
        with lock:
            return adapter.get_results(keyed_tokens_hits, done)

    return functools.update_wrapper(wrapper, fn)


def _async_memoize_batched[T, R](  # noqa: C901
    fn: _AsyncBatchedFn[T, R], adapter: _Adapter[T, R]
) -> _AsyncBatchedFn[T, R]:
    running = WeakValueDictionary[Hashable, asyncio.Future[R]]()
    queue: dict[Hashable, _AsyncJob[T, R]] = {}

    async def _dispatch() -> None:
        jobs = {**queue}
        queue.clear()

        try:
            values = [*await fn([job.token for job in jobs.values()])]

            if len(values) != len(jobs):
                msg = (
                    'Input batch size is not equal to output: '
                    f'{len(values)} != {len(jobs)}'
                )
                raise RuntimeError(msg)  # noqa: TRY301

        except BaseException as exc:  # noqa: BLE001
            for key, job in jobs.items():
                running.pop(key)
                job.future.set_exception(exc)

        else:
            for job, value in zip(jobs.values(), values):
                job.future.set_result(value)

    async def _load(key: Hashable, token: T) -> R:
        if f := running.get(key):
            return await f

        running[key] = f = asyncio.Future[R]()
        queue[key] = _AsyncJob(token, f)
        if len(queue) == 1:
            await _dispatch()

        return await f

    async def wrapper(tokens: Iterable[T]) -> list[R]:
        keyed_tokens_hits, misses = adapter.merge_inputs(tokens)
        rs = await asyncio.gather(*(_load(k, t) for k, t in misses.items()))

        # Process misses
        done = dict(zip(misses, rs))
        return adapter.get_results(keyed_tokens_hits, done)

    return functools.update_wrapper(wrapper, fn)


# ------------------------------- decorations --------------------------------


def _memoize[**P, R](
    fn: Callable[P, R],
    *,
    cache: _AbstractCache | None = None,
    key_fn: _KeyFn = make_key,
    batched: bool,
) -> Callable[P, R]:
    if batched:
        adapter = _Adapter(key_fn, cache)

        if iscoroutinefunction(fn):
            w = cast(
                'Callable[P, R]',
                _async_memoize_batched(
                    cast('_AsyncBatchedFn', fn), adapter=adapter
                ),
            )
        else:
            w = cast(
                'Callable[P, R]',
                _sync_memoize_batched(cast('_BatchedFn', fn), adapter=adapter),
            )

    elif iscoroutinefunction(fn):
        w = cast('Callable[P, R]', _async_memoize(fn, key_fn, cache))

    else:
        w = _sync_memoize(fn, key_fn, cache)

    w.cache = cache  # type: ignore[attr-defined]
    return w


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
