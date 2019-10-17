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
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from .size_hint import make_sized

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


def _chunk_size_hint(iterable, size, *_, **__):
    return len(range(0, len(iterable), size))


@make_sized(hint=_chunk_size_hint)
def chunked(iterable: Iterable[T],
            size: int,
            lazy: bool = False) -> Iterator[Iterable[T]]:
    """Yields chunks of at most `size` items from iterable"""
    fn = _chunked_lazy if lazy else _chunked_greedy
    return fn(iterable, size)


@make_sized(hint=_chunk_size_hint)
def sliced(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yields slices of at most `size` items from iterable"""
    return (seq[offset: offset + size] for offset in range(0, len(seq), size))


@make_sized(hint=lambda iterable, size: len(iterable) + 1 - size)
def windowed(iterable: Iterable[T], size: int) -> Iterator[Tuple[T]]:
    """windowed('ABCDEFG') --> ABC BCD CDE DEF EFG"""
    return zip(*(itertools.islice(it, ahead, None)
                 for ahead, it in enumerate(itertools.tee(iterable, size))))


def iter_none(fn: Callable[[], T],
              marker: Any = None) -> Iterator[T]:
    """Yields `fn()` until it is `marker`"""
    return itertools.takewhile(lambda r: r is not marker,
                               itertools.starmap(fn, itertools.repeat(())))


def eat(iterable: Iterable) -> None:
    collections.deque(iterable, maxlen=0)
