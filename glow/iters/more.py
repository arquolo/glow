__all__ = (
    'as_iter',
    'chunked',
    'eat',
    'iter_none',
    'sliced',
    'windowed',
)

import collections
import itertools
from collections import abc
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar('T')


def as_iter(obj: Union[Iterable[T], T, None],
            times: Optional[int] = None,
            base=abc.Iterable) -> Iterable[T]:
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, base):
        return obj
    return itertools.repeat(obj, times=times)


@dataclass
class SizedSequence(Generic[T]):
    iterator: Iterable[T]
    total: int = 0

    def __iter__(self) -> Iterator[T]:
        yield from self.iterator

    def __len__(self) -> int:
        return self.total


def _chunked_greedy(iterable: Iterable[T],
                    size: int) -> Iterator[Tuple[T]]:
    iterator = iter(iterable)
    return iter(lambda: tuple(itertools.islice(iterator, size)),
                ())


def _chunked_lazy(iterable: Iterable[T],
                  size: int) -> Iterator[Iterable[T]]:
    iterator = iter(iterable)
    marker = object()
    while True:
        # Check to see whether we're at the end of the source iterable
        item = next(iterator, marker)
        if item is marker:
            return

        # Clone the source and yield an n-length slice
        iterator, it = itertools.tee(itertools.chain([item], iterator))
        yield itertools.islice(it, size)

        # Advance the source iterable
        next(itertools.islice(iterator, size, size), None)


def chunked(iterable: Iterable[T],
            size: int,
            lazy: bool = False) -> Iterator[Iterable[T]]:
    """Yields chunks of at most `size` items from iterable"""
    fn = _chunked_lazy if lazy else _chunked_greedy
    iterator = fn(iterable, size)
    try:
        chunks, remainder = divmod(len(iterable), size)
        return SizedSequence(iterator, total=chunks + bool(remainder))
    except TypeError:
        return iterator


def sliced(iterable: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yields slices of at most `size` items from iterable"""
    return (iterable[offset: offset + size]
            for offset in range(0, len(iterable), size))


def windowed(iterable: Iterable[T], size: int) -> Iterator[Tuple[T]]:
    """windowed('ABCDEFG') --> ABC BCD CDE DEF EFG"""
    return zip(*(itertools.islice(it, ahead, None)
                 for ahead, it in enumerate(itertools.tee(iterable, size))))


def iter_none(fn: Callable[[], T],
              marker: Any = None) -> Iterator[T]:
    return itertools.takewhile(lambda r: r is not marker,
                               itertools.starmap(fn, itertools.repeat(())))


def eat(iterable: Iterable) -> None:
    collections.deque(iterable, maxlen=0)
