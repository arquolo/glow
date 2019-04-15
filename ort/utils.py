import itertools
from collections import Counter
from collections.abc import Iterable

from wrapt import decorator


def unique(name, _names=Counter()):  # ! "_names" is mutable for the reason
    """Returns unique string"""
    if name is None:
        return None
    _names[name] += 1
    return f'{name}_{_names[name]}'


@decorator
def once_per_instance(wrapped, instance, args, kwargs):
    instance.__dict__.setdefault('__results__', dict())
    if wrapped not in instance.__results__:
        instance.__results__[wrapped] = wrapped(*args, **kwargs)
    return instance.__results__[wrapped]


def as_iter(obj, times=None, base=Iterable):
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, base):
        return obj
    return itertools.repeat(obj, times=times)


def grouped(iterable, size):
    """Yield groups of `size` items from iterator"""
    iterator = iter(iterable)
    yield from iter(lambda: list(itertools.islice(iterator, size)),
                    [])


def pdict(d: dict, sep=', '):
    return sep.join(f'{key!s}={value!r}' for key, value in d.items())


def pbytes(x: int):
    prefixes = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi')
    for prefix in prefixes:
        if x < 1024:
            return f'{x:.2f} {prefix}B'
        x = x / 1024
    return f'{x:.2f} YiB'
