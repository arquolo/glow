__all__ = [
    'as_iter', 'chunked', 'eat', 'ichunked', 'iter_none', 'sliced', 'windowed'
]

import collections
import collections.abc
import enum
import threading
from itertools import chain, count, islice, repeat, starmap, takewhile, tee
from typing import (Any, Callable, Iterable, Iterator, Optional, Sequence,
                    TypeVar, Union, overload)

from .len_helpers import SizedIter, as_sized

_T = TypeVar('_T')
_U = TypeVar('_U')


class _Empty(enum.Enum):
    token = 0


def as_iter(obj: Union[Iterable[_T], _T, None],
            times: int = None) -> Iterable[_T]:
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, collections.abc.Iterable):
        return islice(obj, times)
    return repeat(obj) if times is None else repeat(obj, times)


def windowed(it: Iterable[_T], size: int) -> Iterator[Sequence[_T]]:
    """
    >>> [*windowed(range(5), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    """
    iters = tee(it, size)
    slices = map(islice, iters, count(), repeat(None))
    return zip(*slices)


def sliced(it: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    """
    Yields slices of at most `size` items from `sequence`.

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
    """
    Iterates over `iterable` packing consecutive items to chunks
    with size at most of `size`.

    >>> s = chunked(range(10), 3)
    >>> len(s)
    4
    >>> [*s]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    chunks = map(islice, repeat(iter(it)), repeat(size))
    return iter(map(tuple, chunks).__next__, ())  # type: ignore


@as_sized(hint=chunk_hint)
def ichunked(it: Iterable[_T], size: int) -> Iterator[Iterator[_T]]:
    """
    Iterates over `iterable` packing consecutive items to chunks
    with size at most of `size`. Yields iterators.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> chunks = [*s]
    >>> [(*chunk, ) for chunk in chunks]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    iter_ = iter(it)
    # while (item := next(iter_, _Empty.token)) is not _Empty.token:  # ! py38
    while True:
        item: Union[_T, _Empty] = next(iter_, _Empty.token)
        if item is _Empty.token:  # check that end reached
            return

        iter_, rest = tee(chain([item], iter_))  # clone source
        yield SizedIter(islice(iter_, size), size)

        iter_ = islice(rest, size, None)


@overload
def iter_none(fn: Callable[[], Optional[_T]],
              empty: None = ...) -> Iterator[_T]:
    ...


@overload
def iter_none(fn: Callable[[], Union[_T, _U]],
              empty: _U = ...) -> Iterator[_T]:
    ...


def iter_none(fn, empty=None):
    """Yields `fn()` until it is `marker`"""
    return takewhile(lambda r: r is not empty, starmap(fn, repeat(())))


def eat(iterable: Iterable[Any], async_: bool = False) -> None:
    """Consume `iterable`, if `async_` then in background thread"""
    if not async_:
        collections.deque(iterable, maxlen=0)
    else:
        threading.Thread(
            target=collections.deque, args=(iterable, 0), daemon=True).start()
