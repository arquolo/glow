__all__ = [
    'as_iter', 'chunked', 'eat', 'groupby', 'ichunked', 'ilen', 'roundrobin',
    'windowed'
]

import threading
from collections import deque
from collections.abc import (Callable, Hashable, Iterable, Iterator, Mapping,
                             Sequence, Sized)
from itertools import batched, chain, count, cycle, islice, repeat
from typing import Protocol, overload


class SupportsSlice[T](Sized, Protocol):
    def __getitem__(self, __s: slice) -> T:
        ...


# ----------------------------------------------------------------------------


def as_iter[T](obj: Iterable[T] | T, limit: int | None = None) -> Iterator[T]:
    """Make iterator with at most `limit` items"""
    if isinstance(obj, Iterable):
        return islice(obj, limit)
    return repeat(obj) if limit is None else repeat(obj, limit)


# ----------------------------------------------------------------------------


def _dispatch(fallback_fn, fn, it, *args):
    if (not isinstance(it, Sized) or not hasattr(it, '__getitem__')
            or isinstance(it, Mapping)):
        return fallback_fn(it, *args)

    r = fn(it, *args)
    if isinstance(it, Sequence):
        return r

    try:
        # Ensure that slice is supported by prefetching 1st item
        first_or_none = *islice(r, 1),
    except TypeError:
        return fallback_fn(it, *args)
    else:
        return chain(first_or_none, r)


# ----------------------------------------------------------------------------


def window_hint(it, size):
    return len(it) + 1 - size


def chunk_hint(it, size):
    return len(range(0, len(it), size))


def _sliced_windowed[T](s: SupportsSlice[T], size: int) -> Iterator[T]:
    indices = range(len(s) + 1)
    slices = map(slice, indices[:-size], indices[size:])
    return map(s.__getitem__, slices)


def _windowed[T](it: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    if size == 1:  # Trivial case
        return zip(it)

    it = iter(it)
    w = deque(islice(it, size), maxlen=size)

    if len(w) != size:
        return iter(())
    return map(tuple, chain([w], map(w.__iadd__, zip(it))))


def _sliced[T](s: SupportsSlice[T], size: int) -> Iterator[T]:
    indices = range(len(s) + size)
    slices = map(slice, indices[::size], indices[size::size])
    return map(s.__getitem__, slices)


def _chunked[T](it: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    if size == 1:  # Trivial case
        return zip(it)

    return batched(it, size)


# ---------------------------------------------------------------------------


@overload
def windowed[T](it: SupportsSlice[T], size: int) -> Iterator[T]:
    ...


@overload
def windowed[T](it: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    ...


def windowed(it, size):
    """Retrieve overlapped windows from iterable.
    Tries to use slicing if possible.

    >>> [*windowed(range(6), 3)]
    [range(0, 3), range(1, 4), range(2, 5), range(3, 6)]

    >>> [*windowed(iter(range(6)), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    return _dispatch(_windowed, _sliced_windowed, it, size)


@overload
def chunked[T](__it: SupportsSlice[T], size: int) -> Iterator[T]:
    ...


@overload
def chunked[T](__it: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    ...


def chunked(it, size):
    """
    Splits iterable to chunks of at most size items each.
    Uses slicing if possible.
    Each next() on result will advance passed iterable to size items.

    >>> [*chunked(range(10), 3)]
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]

    >>> [*chunked(iter(range(10)), 3)]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    return _dispatch(_chunked, _sliced, it, size)


# ----------------------------------------------------------------------------


def _deiter[T](q: deque[T]) -> Iterator[T]:
    # Same as iter_except(q.popleft, IndexError) from docs of itertools
    try:
        while True:
            yield q.popleft()
    except IndexError:
        return


def ichunked[T](it: Iterable[T], size: int) -> Iterator[Iterator[T]]:
    """Split iterable to chunks of at most size items each.

    Does't consume items from passed iterable to return complete chunk
    unlike chunked, as yields iterators, not sequences.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> [[*chunk] for chunk in s]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    if size == 1:  # Trivial case
        yield from map(iter, zip(it))
        return

    it = iter(it)
    while head := deque(islice(it, 1)):
        # Remaining chunk items
        body = islice(it, size - 1)

        # Cache for not-yet-consumed
        tail = deque[T]()

        # Include early fetched item into chunk
        yield chain(_deiter(head), body, _deiter(tail))

        # Advance and fill internal cache, expand tail with items from body
        tail += body


# ----------------------------------------------------------------------------


def ilen(iterable: Iterable) -> int:
    """Return number of items in *iterable*.

    This consumes iterable, so handle with care.
    See `more_itertools.ilen`.
    """
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)


def eat(iterable: Iterable, daemon: bool = False) -> None:
    """Consume iterable, daemonize if needed (move to background thread)"""
    if daemon:
        threading.Thread(target=deque, args=(iterable, 0), daemon=True).start()
    else:
        deque(iterable, 0)  # Same as `more_itertools.consume(..., n=None)`


def roundrobin[T](*iterables: Iterable[T]) -> Iterator[T]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    iters = cycle(iter(it) for it in iterables)
    for pending in range(len(iterables) - 1, -1, -1):
        yield from map(next, iters)
        iters = cycle(islice(iters, pending))


# ----------------------------------------------------------------------------


@overload
def groupby[T, K: Hashable](iterable: Iterable[T], /,
                            key: Callable[[T], K]) -> dict[K, list[T]]:
    ...


@overload
def groupby[T, K: Hashable, V](iterable: Iterable[T], /, key: Callable[[T], K],
                               value: Callable[[T], V]) -> dict[K, list[V]]:
    ...


def groupby(iterable, /, key, value=lambda x: x):
    """Group items from iterable by key.

    >>> groupby([True, (), 1, 0], bool)
    {True: [True, 1], False: [(), 0]}

    """
    r: dict = {}
    for x in iterable:
        r.setdefault(key(x), []).append(value(x))
    return r
