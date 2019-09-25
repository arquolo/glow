__all__ = ('sizeof', )

import sys
from collections.abc import Collection
from inspect import isgetsetdescriptor, ismemberdescriptor


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
