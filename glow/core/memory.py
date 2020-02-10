__all__ = ('sizeof', 'Size')

import sys
import functools
from collections import abc
from inspect import isgetsetdescriptor, ismemberdescriptor

import wrapt


# TODO: replace `wrapt.ObjectProxy` with `int`
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


def for_unseen(fn, default=Size):
    def wrapper(obj, seen=None):
        if seen is None:
            seen = set()
        id_ = id(obj)
        if id_ in seen:
            return default()

        seen.add(id_)
        return fn(obj, seen)

    return functools.update_wrapper(wrapper, fn)


@functools.singledispatch
@for_unseen
def sizeof(obj, seen=None) -> Size:
    """
    Computes size of object, no matter how complex it is

    Inspired by
    [PySize](https://github.com/bosswissam/pysize/blob/master/pysize.py)
    """
    size = Size(sys.getsizeof(obj))

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
        size += sum((sizeof(k, seen) for k in obj.keys()), Size())
        size += sum((sizeof(v, seen) for v in obj.values()), Size())
    elif isinstance(obj, abc.Collection):
        size += sum((sizeof(item, seen) for item in obj), Size())

    if hasattr(obj, '__slots__'):
        size += sum((sizeof(getattr(obj, slot, None), seen=seen)
                     for class_ in type(obj).mro()
                     for slot in getattr(class_, '__slots__', ())), Size())
    return size


@wrapt.when_imported('numpy')
def _numpy(numpy):
    @sizeof.register(numpy.ndarray)
    @for_unseen
    def _sizeof(obj, seen=None) -> Size:
        return Size(max(sys.getsizeof(obj), obj.nbytes))


@wrapt.when_imported('torch')
def _torch(torch):
    @sizeof.register(torch.Tensor)
    @for_unseen
    def _sizeof(obj, seen=None) -> Size:
        size = sys.getsizeof(obj)
        if not obj.is_cuda:
            size += obj.numel() * obj.element_size()
        return Size(size)  # TODO: test, maybe useless when grads are attached
