__all__ = ('memoize', )

import argparse
import functools
import threading
import weakref
from typing import (Callable, Counter, Dict, Generic, MutableMapping, Type,
                    TypeVar, cast)

from typing_extensions import Literal

from ..memory import Size, sizeof
from .concurrency import interpreter_lock

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_Policy = Literal['raw', 'lru', 'mru']


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
    store: Dict[str, Record[_T]]
    stats: Stats
    size: Size
    capacity: Size

    def keys(self):
        return self.store.keys()

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, key: str) -> _T:
        """Retrieve value from cache"""
        return self.store[key].value

    def __setitem__(self, key: str, value: _T) -> None:
        record = Record(value)
        if self.evict(record.size):
            self.store[key] = record
            self.size += record.size

    def evict(self, size: int) -> bool:
        """Try to release `size` bytes from storage. Return True on success"""
        raise NotImplementedError


class _CacheBase(_CacheAbc[_T]):
    refs: MutableMapping[int, '_CacheBase'] = weakref.WeakValueDictionary()

    def __init__(self, capacity: int) -> None:
        self.capacity = Size(capacity)
        self.size = Size()
        self.store: Dict[str, Record[_T]] = {}
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
    def evict(self, size: int) -> bool:
        return self.size + size <= self.capacity


class _LruCache(_CacheBase[_T]):
    drop_recent = False

    def __getitem__(self, key: str) -> _T:
        self.store[key] = record = self.store.pop(key)
        return record.value

    def evict(self, size: int) -> bool:
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


def _memoize(cache: _CacheBase, fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'
        try:
            with cache.lock:
                value = cache[key]
                cache.stats.hits += 1
        except KeyError:
            value = fn(*args, **kwargs)
            with cache.lock:
                cache.stats.misses += 1
                cache[key] = value
        return value  # noqa: R504

    wrapper.cache = cache  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, fn))


def memoize(capacity: int, policy: _Policy = 'raw') -> Callable[[_F], _F]:
    """Returns dict-cache decorator.

    Arguments:
      - `capacity` - size in bytes
      - `policy` - eviction policy, one of (`"raw"`, `"lru"`, `"mru"`)

    """
    rtype = Callable[[_F], _F]
    if not capacity:
        return cast(rtype, (lambda fn: fn))

    caches: Dict[str, Type[_CacheBase]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    cache_cls = caches.get(policy)
    if cache_cls is None:
        raise ValueError(
            f'Unknown policy: "{policy}". Only "{set(caches)}" are available')

    res = functools.partial(_memoize, cache_cls(capacity))
    return cast(rtype, res)
