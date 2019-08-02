__all__ = 'CacheAbc', 'CacheLRU', 'Cache'

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from weakref import WeakValueDictionary

from wrapt import FunctionWrapper

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
        def wrapper(fn, _, args, kwargs):
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

        return FunctionWrapper(fn, wrapper, bool(self.capacity))

    def __repr__(self):
        with self._lock:
            line = (f'{self.__class__.__name__}' +
                    f'(items={len(self)},' +
                    f' used={decimate(self.size)}B,' +
                    f' total={decimate(self.capacity)}B)')
            if any(self._stats.values()):
                line += f'({repr_as_obj(self._stats)})'
                self._stats.clear()
            return line

    @classmethod
    def status(cls) -> str:
        with cls._shared_lock:
            return '\n'.join(f'0x{addr:x}: {value!r}'
                             for addr, value in sorted(cls._refs.items()))


class Cache(CacheAbc, dict):
    def can_fit(self, record: Record):
        return self.size + record.size <= self.capacity

    def get(self, key):
        return self[key]

    def put(self, key, value):
        record = Record(value)
        if self.can_fit(record):
            self[key] = value
            self.size += record.size


class CacheLRU(CacheAbc, OrderedDict):
    def can_fit(self, record: Record):
        if record.size > self.capacity:
            return False

        while self.size + record.size > self.capacity:
            self.size -= self.popitem(last=False)[1].size
            self._stats['dropped'] += 1
        return True

    def get(self, key):
        # TODO in python3.8: `return (self[key] := self.pop(key)).value`
        self[key] = record = self.pop(key)
        return record.value

    def put(self, key, value):
        record = Record(value)
        if self.can_fit(record):
            self[key] = record
            self.size += record.size
