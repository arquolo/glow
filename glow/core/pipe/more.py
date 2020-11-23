__all__ = ['as_iter', 'chunked', 'eat', 'ichunked', 'sliced', 'windowed']

import collections
import collections.abc
import enum
import threading
from itertools import chain, count, islice, repeat, tee
from typing import Any, Iterable, Iterator, Sequence, TypeVar, Union

from .len_helpers import SizedIter, as_sized


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_empty = _Empty.token


def as_iter(obj: Union[Iterable[_T], _T, None],
            times: int = None) -> Iterable[_T]:
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, collections.abc.Iterable):
        return islice(obj, times)
    return repeat(obj) if times is None else repeat(obj, times)


def windowed(it: Iterable[_T], size: int) -> Iterator[Sequence[_T]]:
    """Retrieve overlapped windows from iterable.

    >>> [*windowed(range(5), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    """
    iters = tee(it, size)
    slices = map(islice, iters, count(), repeat(None))
    return zip(*slices)


    """Split sequence to slices of at most size items each.

    >>> s = sliced(range(10), 3)
    >>> len(s)
    4
    >>> [*s]
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]
    """
    offsets = range(len(it) + size)
    slices = map(slice, offsets[0::size], offsets[size::size])
    return map(it.__getitem__, slices)


def chunk_hint(it, size):
    return len(range(0, len(it), size))


@as_sized(hint=chunk_hint)
def chunked(it: Iterable[_T], size: int) -> Iterator[Sequence[_T]]:
    """Split iterable to chunks of at most size items each.
    Each next() on result will advance passed iterable to size items.

    >>> s = chunked(range(10), 3)
    >>> len(s)
    4
    >>> [*s]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    chunks = map(islice, repeat(iter(it)), repeat(size))
    return iter(map(tuple, chunks).__next__, ())  # type: ignore


@as_sized(hint=chunk_hint)
    """Split iterable to chunks of at most size items each.

    Does't consume items from passed iterable to return complete chunk
    unlike chunked, as yields iterators, not sequences.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> [[*chunk] for chunk in s]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    iter_ = iter(it)
    while (item := next(iter_, _empty)) is not _empty:
        iter_, rest = tee(chain([item], iter_))  # clone source
        yield SizedIter(islice(iter_, size), size)

        iter_ = islice(rest, size, None)


def eat(iterable: Iterable[Any], async_: bool = False) -> None:
    """Consume `iterable`, if `async_` then in background thread"""
    if not async_:
        collections.deque(iterable, maxlen=0)
    else:
        threading.Thread(
            target=collections.deque, args=(iterable, 0), daemon=True).start()
