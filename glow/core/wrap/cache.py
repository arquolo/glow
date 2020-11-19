__all__ = ['memoize']

import argparse
import functools
import threading
import weakref
from collections import Counter
from typing import (Callable, Dict, Generic, Hashable, Literal, MutableMapping,
                    Type, TypeVar, cast)

from .._repr import Si
from .._sizeof import sizeof
from .concurrency import interpreter_lock

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]


class Record(Generic[_T]):
    __slots__ = ('value', 'size')

    def __init__(self, value: _T) -> None:
        self.value = value
        self.size = sizeof(value)

    def __repr__(self) -> str:
        return f'({self.value} / {self.size})'


class Stats(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__ = Counter()
        self.__dict__.update(**kwargs)

    def __getattr__(self, name: str) -> int:
        return self.__dict__[name]


class _CacheAbc(Generic[_T]):
    store: Dict[Hashable, Record[_T]]
    stats: Stats
    size: Si
    capacity: Si

    def keys(self):
        return self.store.keys()

    def clear(self):
        self.store.clear()
        self.size = Si.bits()

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, key: Hashable) -> _T:
        """Retrieve value from cache"""
        return self.store[key].value

    def __setitem__(self, key: Hashable, value: _T) -> None:
        record = Record(value)
        if self.evict(record.size):
            self.store[key] = record
            self.size += record.size

    def evict(self, size: Si) -> bool:
        """Try to release `size` bytes from storage. Return True on success"""
        raise NotImplementedError


class _CacheBase(_CacheAbc[_T]):
    refs: MutableMapping[int, '_CacheBase'] = weakref.WeakValueDictionary()

    def __init__(self, capacity: int) -> None:
        self.capacity = Si.bits(capacity)
        self.size = Si.bits()
        self.store: Dict[Hashable, Record[_T]] = {}
        self.stats = Stats()
        self.lock = threading.RLock()
        self.refs[id(self)] = self

    def __repr__(self) -> str:
        with self.lock:
            line = (
                f'{type(self).__name__}'
                f'(items={len(self)}, used={self.size}, total={self.capacity})'
            )
            if any(vars(self.stats).values()):
                line += f'-{self.stats}'
            return line

    @classmethod
    def status(cls) -> str:
        with interpreter_lock():
            return '\n'.join(f'{id_:x}: {value!r}'
                             for id_, value in sorted(cls.refs.items()))


class _HeapCache(_CacheBase[_T]):
    def evict(self, size: Si) -> bool:
        return self.size + size <= self.capacity


class _LruCache(_CacheBase[_T]):
    drop_recent = False

    def __getitem__(self, key: Hashable) -> _T:
        self.store[key] = record = self.store.pop(key)
        return record.value

    def evict(self, size: Si) -> bool:
        if size > self.capacity:
            return False

        while self.size + size > self.capacity:
            if self.drop_recent:
                self.size -= self.store.popitem()[1].size
            else:
                self.size -= self.store.pop(next(iter(self.store))).size
            self.stats.dropped += 1
        return True


class _MruCache(_LruCache[_T]):
    drop_recent = True


def _memoize(cache: _CacheBase, key_fn: _KeyFn, fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = key_fn(*args, **kwargs)
        try:
            with cache.lock:
                value = cache[key]
                cache.stats.hits += 1
        except KeyError:
            try:
                value = fn(*args, **kwargs)
            except BaseException as exc:
                exc.__context__ = None
                raise
            with cache.lock:
                cache.stats.misses += 1
                cache[key] = value
        return value  # noqa: R504

    wrapper.cache = cache  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, fn))


def _key_fn(*args, **kwargs) -> str:
    return f'{args}{kwargs}'


def memoize(capacity: int,
            policy: _Policy = 'raw',
            key_fn: _KeyFn = _key_fn) -> Callable[[_F], _F]:
    """Returns dict-cache decorator.

    Parameters:
      - capacity - size in bytes
      - policy - eviction policy, one of "raw", "lru" or "mru"
    """
    if not capacity:
        return lambda fn: fn

    caches: Dict[str, Type[_CacheBase]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    cache_cls = caches.get(policy)
    if cache_cls is None:
        raise ValueError(
            f'Unknown policy: "{policy}". Only "{set(caches)}" are available')

    return functools.partial(_memoize, cache_cls(capacity), key_fn)
