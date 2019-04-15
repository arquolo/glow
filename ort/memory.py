import itertools
import sys
from collections import Counter, OrderedDict
from inspect import isgetsetdescriptor, ismemberdescriptor
from threading import RLock
from weakref import WeakValueDictionary

try:
    import numpy
except ImportError:
    numpy = None
try:
    import torch
except ImportError:
    torch = None
from wrapt import FunctionWrapper

from .utils import pdict, pbytes
from .debug import sprint


def sizeof(obj, seen=None):
    """
    Computes size of object, no matter how complex it is

    Inspired by
    [PySize](https://github.com/bosswissam/pysize/blob/master/pysize.py)
    """
    if seen is None:
        seen = set()
    id_ = id(obj)
    if id_ in seen:
        return 0

    seen.add(id_)
    size = sys.getsizeof(obj)
    if numpy is not None and isinstance(obj, numpy.ndarray):
        return max(size, obj.nbytes)
    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.device.type == 'cpu':
            size += obj.numel() * obj.element_size()
        return size

    # protection from self-referencing
    if hasattr(obj, '__dict__'):
        for d in (cls.__dict__['__dict__']
                  for cls in type(obj).__mro__ if '__dict__' in cls.__dict__):
            if isgetsetdescriptor(d) or ismemberdescriptor(d):
                size += sizeof(obj.__dict__, seen=seen)
            break

    if isinstance(obj, dict):
        size += sum(sizeof(k, seen) + sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (str, bytes, bytearray)):
        pass
    elif hasattr(obj, '__iter__'):
        size += sum(sizeof(item, seen=seen) for item in obj)

    if hasattr(obj, '__slots__'):
        size += sum(sizeof(getattr(obj, slot), seen=seen)
                    for slot in obj.__slots__ if hasattr(obj, slot))
    return size


class _Record:
    __slots__ = 'value', 'size'

    def __init__(self, value):
        self.value = value
        self.size = sizeof(value)


class CacheAbc:
    _counter = itertools.count()
    _lock = RLock()
    _refs = WeakValueDictionary()

    def __init__(self, capacity, lock=None):
        self.capacity = int(capacity)
        self._lock = lock = lock or RLock()
        self._size = 0
        self._stats = Counter()
        self._refs[next(self._counter)] = self

        def wrapper(wrapped, _, args, kwargs):
            key = f'{wrapped}{args or ""}{kwargs or ""}'
            try:
                with lock:
                    value = self.get(key)
                    self._stats['hits'] += 1
            except KeyError:
                value = wrapped(*args, **kwargs)
                with lock:
                    self._stats['misses'] += 1
                    self.put(key, value)
            return value

        self._wrapper = wrapper

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def __call__(self, wrapped):
        return FunctionWrapper(wrapped, self._wrapper, bool(self.capacity))

    def __repr__(self):
        with self._lock:
            line = (f'{self.__class__.__name__}'
                    f'(items={len(self)},'
                    f' used={pbytes(self._size)},'
                    f' total={pbytes(self.capacity)})')
            if any(self._stats.values()):
                line += f'({pdict(self._stats)})'
                self._stats.clear()
            return line

    @classmethod
    def info_all(cls, *_):
        with cls._lock:
            for key, value in sorted(cls._refs.items()):
                sprint(f' [info] {key!s}: {value!r}')


class Cache(CacheAbc, dict):
    def get(self, key):
        return self[key]

    def put(self, key, value):
        size = sizeof(value)
        if self._size + size < self.capacity:
            self[key] = value
            self._size += size


class CacheLRU(CacheAbc, OrderedDict):
    def shrink(self, desired=0):
        if self.capacity < desired:
            return -1

        drops = 0
        while self.capacity < self._size + desired:
            self._size -= self.popitem(last=False)[1].size
            drops += 1
        return drops

    def get(self, key):
        self[key] = record = self.pop(key)
        return record.value

    def put(self, key, value):
        record = _Record(value)
        dropped = self.shrink(desired=record.size)
        if dropped >= 0:
            self[key] = record
            self._size += record.size
            self._stats['dropped'] += dropped
