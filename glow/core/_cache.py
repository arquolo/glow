"""
Available functions:

- _cache_temporal/_ts_mux (old shared_call)
    - call(*args, **kwargs):
        key = key_fn(*args, **kwargs)
        if future := _get_future(key):
            return await future
        else:
            future = Future()
            _set_future(key, future)
            submit(Job(future, call_impl, *args, **kwargs))
            return await future

- _cache_spatial (old memoize)
    - call(*args, **kwargs):
        key = key_fn(*args, **kwargs)
        if (item := storage[*args, **kwargs]) is not empty:
            return value
        else:
            storage[*args, **kwargs] = value = call_impl()
            return value

"""

from __future__ import annotations

__all__ = ['cache', 'threadlocal']

import enum
import functools
import sys
import time
from collections.abc import Hashable, KeysView, MutableMapping, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from queue import Empty, SimpleQueue
from threading import Lock, RLock, Thread, local
from typing import Any, Callable, ClassVar, Generic, Literal, TypeVar, cast
from weakref import WeakValueDictionary

from ._repr import si_bin
from ._sizeof import sizeof

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_ZeroArgsF = TypeVar('_ZeroArgsF', bound=Callable[[], Any])
_BatchF = TypeVar('_BatchF', bound=Callable[[Sequence], list])
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]


class _Empty(enum.Enum):
    token = 0


_empty = _Empty.token

# --------------------------------- helpers ---------------------------------

_KWD_MARK = object()


class _HashedSeq(list):  # List is mutable, that's why not NamedTuple
    __slots__ = 'hashvalue'

    def __init__(self, tup: tuple):
        self[:] = tup
        self.hashvalue = hash(tup)  # Memorize hash

    def __hash__(self):
        return self.hashvalue


def _make_key(*args, **kwargs) -> _HashedSeq:
    """Copied from functools._make_key, as private function"""
    key = args
    if kwargs:
        key += _KWD_MARK,
        for item in kwargs.items():
            key += item
    if len(key) == 1 and type(key[0]) in (int, str):
        return key[0]
    return _HashedSeq(key)


@dataclass(frozen=True)
class _Job(Generic[_T]):
    item: _T
    future: Future[_T] = field(default_factory=Future)


def _dispatch(func: Callable[[Sequence], list], jobs: Sequence[_Job]) -> None:
    try:
        results = func([job.item for job in jobs])
        assert len(results) == len(jobs)

    except BaseException as exc:  # noqa: PIE786
        for job in jobs:
            job.future.set_exception(exc)

    else:
        for job, res in zip(jobs, results):
            job.future.set_result(res)


class _DeferredStack(ExitStack):
    """
    ExitStack that allows deferring.
    When return value of callback function should be accessible, use this.
    """
    def defer(self, fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
        future: Future[_T] = Future()

        def apply(future: Future[_T]) -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: PIE786
                future.set_exception(exc)
            else:
                future.set_result(result)

        self.callback(apply, future)
        return future


@contextmanager
def _interpreter_lock(timeout=1_000):
    """
    Prevents thread switching in underlying scope, thus makes it completely
    thread-safe. Although adds high performance penalty.

    See tests for examples.
    """
    with ExitStack() as stack:
        stack.callback(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(timeout)
        yield


# -------------------------- zero argument wrappers --------------------------


def call_once(fn: _ZeroArgsF) -> _ZeroArgsF:
    """Makes `fn()` callable a singleton"""
    lock = RLock()

    def wrapper():
        with _DeferredStack() as stack, lock:
            if fn.__future__ is None:
                # This way setting future is protected, but fn() is not
                fn.__future__ = stack.defer(fn)

        return fn.__future__.result()

    fn.__future__ = None  # type: ignore
    return cast(_ZeroArgsF, functools.update_wrapper(wrapper, fn))


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    """Thread-local singleton factory, mimics `functools.partial`"""
    local_ = local()

    def wrapper() -> _T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return wrapper


# ----------------------------- temporal caching ----------------------------


def _temporal(fn: _F) -> _F:
    """
    Merges simultaneous calls with the same `args` to single one.
    Not suitable for recursive callables.
    """
    access_lock = Lock()
    locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()

    def wrapper(*args, **kwargs):
        key = _make_key(*args, **kwargs)
        with access_lock:
            if (lock := locks.get(key)) is None:
                lock = locks.setdefault(key, Lock())
        with lock:
            return fn(*args, **kwargs)

    return cast(_F, functools.update_wrapper(wrapper, fn))


def _temporal_batch(fn: _BatchF) -> _BatchF:
    """
    Merges simultaneous calls with the same `args` to single one.
    Not suitable for recursive callables.
    """
    access_lock = Lock()
    locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()

    def wrapper(items: Sequence) -> list:
        with access_lock:
            active_locks = {
                item: (lock if (lock := locks.get(item)) is not None else
                       locks.setdefault(item, Lock())) for item in items
            }

        with ExitStack() as stack:
            for lock in active_locks.values():
                stack.enter_context(lock)
            active_items = [*active_locks]
            results = dict(zip(active_items, fn(active_items)))

        return [results[item] for item in items]

    return cast(_BatchF, functools.update_wrapper(wrapper, fn))


# ----------------------------- spatial caching -----------------------------


@dataclass(repr=False)
class _Node(Generic[_T]):
    __slots__ = ('value', 'size')
    value: _T
    size: int

    def __repr__(self) -> str:
        return repr(self.value)


@dataclass(repr=False, eq=False)
class Stats:
    hits: int = 0
    misses: int = 0
    drops: int = 0

    def __bool__(self):
        return any(asdict(self).values())

    def __repr__(self):
        data = ', '.join(f'{k}={v}' for k, v in asdict(self).items() if v)
        return f'{type(self).__name__}({data})'


class _IStore(Generic[_T]):
    def __len__(self) -> int:
        raise NotImplementedError

    def store_clear(self) -> None:
        raise NotImplementedError

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        raise NotImplementedError

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        raise NotImplementedError

    def can_swap(self, size: int) -> bool:
        raise NotImplementedError


@dataclass(repr=False)
class _InitializedStore:
    capacity: int
    size: int = 0
    stats: Stats = field(default_factory=Stats)


@dataclass(repr=False)
class _DictMixin(_InitializedStore, _IStore[_T]):
    lock: RLock = field(default_factory=RLock)

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
            size = int(sizeof(value))
            if (self.size + size <= self.capacity) or self.can_swap(size):
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
            args += [f'stats={self.stats}']
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
    def can_swap(self, size: int) -> bool:
        return False


class _LruCache(_Store[_T]):
    drop_recent = False

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        if node := self.store.pop(key, None):
            self.store[key] = node
            return node
        return None

    def can_swap(self, size: int) -> bool:
        if size > self.capacity:
            return False

        while self.size + size > self.capacity:
            if self.drop_recent:
                self.size -= self.store.popitem()[1].size
            else:
                self.size -= self.store.pop(next(iter(self.store))).size
            self.stats.drops += 1
        return True


class _MruCache(_LruCache[_T]):
    drop_recent = True


# ------------------------------ cache wrappers ------------------------------


def _spatial(cache: _DictMixin, key_fn: _KeyFn, func: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = key_fn(*args, **kwargs)

        if (value := cache[key]) is not _empty:
            return value

        cache[key] = value = func(*args, **kwargs)
        return value

    wrapper.cache = cache  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, func))


def _spatial_batched(cache: _DictMixin, key_fn: _KeyFn,
                     func: _BatchF) -> _BatchF:
    assert callable(func)
    lock = Lock()
    pending: dict[Hashable, _Job] = {}

    # def _safe_dispatch():
    #     with lock:
    #         keys, jobs = zip(*pending.items())
    #         pending.clear()

    #     _dispatch(func, jobs)

    #     for key, job in zip(keys, jobs):
    #         if not job.future.exception():
    #             cache[key] = job.future.result()

    # def _get_future(stack: ExitStack, key: Hashable, item: Any) -> Future:
    #     if job := pending.get(key):
    #         return job.future

    #     future = Future()  # type: ignore

    #     if (value := cache[key]) is not _empty:
    #         future.set_result(value)
    #         return future

    #     pending[key] = _Job(item, future)
    #     if len(pending) == 1:
    #         stack.callback(_safe_dispatch)

    #     return future

    def wrapper(items: Sequence[Hashable]) -> list:

        results = {}
        with lock:
            for item in items:
                if (value := cache[item]) is not _empty:
                    results[item] = value
                elif not pending.get(item):
                    pending[item] = _Job(item)
            jobs = [*pending.values()]
            pending.clear()

        if jobs:
            _dispatch(func, jobs)

        with lock:
            for job in jobs:
                if not job.future.exception():
                    results[job.item] = cache[job.item] = job.future.result()

        return [results[item] for item in items]

        # keyed_items = [(key_fn(item), item) for item in items]
        # with ExitStack() as stack:
        #     with lock:
        #         futs = [_get_future(stack, *ki) for ki in keyed_items]
        # return [fut.result() for fut in futs]

    wrapper.cache = cache  # type: ignore
    return cast(_BatchF, functools.update_wrapper(wrapper, func))


def cache(
    capacity: int,
    *,
    batched: bool = False,
    policy: _Policy = 'raw',
    key_fn: _KeyFn = _make_key
) -> Callable[[_F], _F] | Callable[[_BatchF], _BatchF]:
    """Returns dict-cache decorator.

    Parameters:
    - capacity - size in bytes.
    - policy - eviction policy, either "raw" (no eviction), or "lru"
      (evict oldest), or "mru" (evict most recent).
    """
    if not capacity:
        return lambda fn: fn

    caches: dict[str, type[_Store]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    if cache_cls := caches.get(policy):
        mem_fn = _spatial if batched else _spatial_batched
        return functools.partial(mem_fn, cache_cls(capacity), key_fn)

    raise ValueError(f'Unknown policy: "{policy}". '
                     f'Only "{set(caches)}" are available')


# --------------------------------- batched ---------------------------------


def stream_batched(func=None, *, batch_size, latency=0.1, timeout=20.):
    """
    Delays start of computation up to `latency` seconds
    in order to fill batch to batch_size items and
    send it at once to target function.
    `timeout` specifies timeout to wait results from worker.

    Simplified version of https://github.com/ShannonAI/service-streamer
    """
    if func is None:
        return functools.partial(
            stream_batched,
            batch_size=batch_size,
            latency=latency,
            timeout=timeout)

    assert callable(func)
    inputs = SimpleQueue()

    def _serve_forever():
        while True:
            jobs = []
            end = time.monotonic() + latency
            for _ in range(batch_size):
                try:
                    jobs.append(inputs.get(timeout=end - time.monotonic()))
                except (Empty, ValueError):
                    break

            if jobs:
                _dispatch(func, jobs)
            else:
                time.sleep(0.001)

    def wrapper(items):
        jobs = [_Job(item) for item in items]
        for job in jobs:
            inputs.put(job)
        return [job.future.result(timeout=timeout) for job in jobs]

    Thread(target=_serve_forever, daemon=True).start()
    return functools.update_wrapper(wrapper, func)
