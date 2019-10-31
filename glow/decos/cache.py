__all__ = ('Cache', 'CacheAbc', 'CacheLRU', 'CacheMRU')

import functools
from argparse import Namespace
from collections import Counter, OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any
from weakref import WeakValueDictionary

from ..core import Size, sizeof
from .thread import interpreter_lock


@dataclass
class Record:
    __slots__ = ('value', 'size')

    def __init__(self, value):
        self.value = value
        self.size = sizeof(value)


class Stats(Namespace):
    def __init__(self, **kwargs):
        self.__dict__ = Counter()
        self.__dict__.update(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return self.__dict__[name]


class CacheAbc:
    _refs = WeakValueDictionary()

    def __init__(self, capacity):
        self.capacity = Size(capacity)
        self.size = Size()
        self._lock = RLock()
        self._stats = Stats()
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
                    self._stats.hits += 1
            except KeyError:
                value = fn(*args, **kwargs)
                with self._lock:
                    self._stats.misses += 1
                    self.put(key, value)
            return value

        wrapper.cache = self
        return wrapper

    def __repr__(self):
        with self._lock:
            line = (
                f'{type(self).__name__}'
                f'(items={len(self)}, used={self.size}, total={self.capacity})'
            )
            if any(vars(self._stats).values()):
                line += f'-{self._stats}'
            return line

    @classmethod
    def status(cls) -> str:
        with interpreter_lock():
            return '\n'.join(f'{hex(addr)}: {value!r}'
                             for addr, value in sorted(cls._refs.items()))


class _DictCache(CacheAbc):
    def put(self, key, value):
        record = Record(value)
        if self.release(record.size):
            self[key] = record
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
            self._stats.dropped += 1
        return True

    def get(self, key):
        # TODO in py3.8: `return (self[key] := self.pop(key)).value`
        self[key] = record = self.pop(key)
        return record.value


class CacheLRU(_OrderedCache):
    drop_recent = False


class CacheMRU(_OrderedCache):
    drop_recent = True
