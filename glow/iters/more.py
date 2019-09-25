__all__ = ('as_iter', 'chunked', 'eat', 'iter_none', 'windowed')

import collections
import itertools
from collections.abc import Iterable


def as_iter(obj, times=None, base=Iterable):
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, base):
        return obj
    return itertools.repeat(obj, times=times)


def chunked(iterable, size):
    """Yields chunks of at most `size` items from iterable"""
    try:
        count = len(iterable)
        return (iterable[init: init + size] for init in range(0, count, size))
    except TypeError:
        iterator = iter(iterable)
        return iter(lambda: list(itertools.islice(iterator, size)),
                    [])


def windowed(iterable, size):
    """windowed('ABCDEFG') --> ABC BCD CDE DEF EFG"""
    return zip(*(itertools.islice(it, ahead, None)
                 for ahead, it in enumerate(itertools.tee(iterable, size))))


def iter_none(fn):
    return itertools.takewhile(lambda r: r is not None,
                               itertools.starmap(fn, itertools.repeat(())))


def eat(iterable):
    collections.deque(iterable, maxlen=0)
