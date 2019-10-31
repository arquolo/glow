__all__ = ('sizeof', 'Size')

import sys
from collections.abc import Collection
from inspect import isgetsetdescriptor, ismemberdescriptor

import wrapt

def sizeof(obj, seen=None):

class Size(wrapt.ObjectProxy):
    """Converts value to prefixed string

    >>> s = Size(2**20)
    >>> s
    Size(1M, base=1024)
    >>> print(s)
    1M
    """
    _suffixes = tuple(enumerate('KMGTPEZY', 1))

    def __init__(self, value: int = 0, base: int = 1024):
        super().__init__(value)
        self._self_base = base

    def __str__(self):
        for order, suffix in reversed(self._suffixes):
            scaled = self.__wrapped__ / self._self_base ** order
            if scaled >= 1:
                return f'{scaled:.4g}{suffix}'
        return f'{self.__wrapped__}'

    def __repr__(self):
        return f'Size({self}, base={self._self_base})'
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

    if ('numpy' in sys.modules
            and isinstance(obj, sys.modules['numpy'].ndarray)):
        return max(size, obj.nbytes)

    if ('torch' in sys.modules
            and sys.modules['torch'].is_tensor(obj)):
        if not obj.is_cuda:
            size += obj.numel() * obj.element_size()
        return size  # TODO: test, maybe useless when grads are attached

    if isinstance(obj, (str, bytes, bytearray)):
        return size

    # protection from self-referencing
    if hasattr(obj, '__dict__'):
        for d in (vars(cl)['__dict__']
                  for cl in obj.__class__.__mro__ if '__dict__' in vars(cl)):
            if isgetsetdescriptor(d) or ismemberdescriptor(d):
                size += sizeof(vars(obj), seen=seen)
            break

    if isinstance(obj, dict):
        size += sum(sizeof(k, seen) + sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, Collection):
        size += sum(sizeof(item, seen=seen) for item in obj)

    if hasattr(obj, '__slots__'):
        size += sum(sizeof(getattr(obj, slot, None), seen=seen)
                    for cl in type(obj).mro()
                    for slot in getattr(cl, '__slots__', ()))
    return size
