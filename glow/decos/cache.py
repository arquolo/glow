__all__ = 'Cache', 'CacheAbc', 'CacheLRU', 'CacheMRU'

import functools
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from weakref import WeakValueDictionary

from ..core import decimate, repr_as_obj, sizeof


@dataclass
class Record:
    value: object
    size: int = field(init=False)

    def __post_init__(self):
        self.size = sizeof(self.value)


@dataclass
class CacheAbc:
    _shared_lock = RLock()
    _refs = WeakValueDictionary()

    capacity: int
    size: int = field(default=0, init=False)
    _lock: RLock = field(default_factory=RLock, init=False)
    _stats: dict = field(default_factory=Counter, init=False)

    def __post_init__(self):
        self._refs[id(self)] = self

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def __call__(self, fn):
        if not self.capacity:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f'{fn}{args}{kwargs}'
            try:
                with self._lock:
                    value = self.get(key)
                    self._stats['hits'] += 1
            except KeyError:
                value = fn(*args, **kwargs)
                with self._lock:
                    self._stats['misses'] += 1
                    self.put(key, value)
            return value

        wrapper.cache = self
        return wrapper

    def __repr__(self):
        with self._lock:
            line = (
                f'{type(self).__name__}' +
                f'(items={len(self)},' +
                ' used={:.4g} {}B,'.format(*decimate(self.size)) +
                ' total={:.4g} {}B)'.format(*decimate(self.capacity))
            )
            if any(self._stats.values()):
                line += f'({repr_as_obj(self._stats)})'
                self._stats.clear()
            return line

    @classmethod
    def status(cls) -> str:
        with cls._shared_lock:
            return '\n'.join(f'{hex(addr)}: {value!r}'
                             for addr, value in sorted(cls._refs.items()))


class _DictCache(CacheAbc):
    def put(self, key, value):
        record = Record(value)
        if self.release(record.size):
            self[key] = value
            self.size += record.size


class Cache(_DictCache, dict):
    def release(self, size):
        return self.size + size <= self.capacity

    def get(self, key):
        return self[key]


class _OrderedCache(_DictCache, OrderedDict):
    def release(self, size):
        if size > self.capacity:
            return False

        while self.size + size > self.capacity:
            self.size -= self.popitem(last=self.drop_recent)[1].size
            self._stats['dropped'] += 1
        return True

    def get(self, key):
        # TODO in python3.8: `return (self[key] := self.pop(key)).value`
        self[key] = record = self.pop(key)
        return record.value


class CacheLRU(_OrderedCache):
    drop_recent = False


class CacheMRU(_OrderedCache):
    drop_recent = True
