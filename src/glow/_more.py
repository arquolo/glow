__all__ = [
    'as_iter',
    'azip',
    'chunked',
    'eat',
    'groupby',
    'ichunked',
    'ilen',
    'roundrobin',
    'windowed',
]

import asyncio
from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from itertools import batched, chain, compress, cycle, islice, repeat
from threading import Thread
from typing import TypeGuard, overload

from ._types import AnyIterable, HasPopleft, SupportsSlice, Unary


def as_iter[T](
    obj: Iterable[T] | T, /, limit: int | None = None
) -> Iterator[T]:
    """Make iterator with at most `limit` items."""
    if isinstance(obj, Iterable):
        return islice(obj, limit)
    return repeat(obj) if limit is None else repeat(obj, limit)


# ----------------------------------------------------------------------------


@overload
def _dispatch[S, *Ts](
    slice_fn: Callable[[SupportsSlice[S], *Ts], Iterator[S]],
    sync_fn: Callable[[Iterable, *Ts], Iterator],
    async_fn: Callable[[AsyncIterable, *Ts], AsyncIterator],
    it: SupportsSlice[S],
    *args: *Ts,
) -> Iterator[S]: ...


@overload
def _dispatch[T, *Ts](
    slice_fn: Callable[[SupportsSlice, *Ts], Iterator],
    sync_fn: Callable[[Iterable[T], *Ts], Iterator[tuple[T, ...]]],
    async_fn: Callable[[AsyncIterable, *Ts], AsyncIterator],
    it: Iterable[T],
    *args: *Ts,
) -> Iterator[tuple[T, ...]]: ...


@overload
def _dispatch[T, *Ts](
    slice_fn: Callable[[SupportsSlice, *Ts], Iterator],
    sync_fn: Callable[[Iterable, *Ts], Iterator],
    async_fn: Callable[[AsyncIterable[T], *Ts], AsyncIterator[tuple[T, ...]]],
    it: AsyncIterable[T],
    *args: *Ts,
) -> AsyncIterator[tuple[T, ...]]: ...


def _dispatch[*Ts](
    slice_fn: Callable[[SupportsSlice, *Ts], Iterator],
    sync_fn: Callable[[Iterable, *Ts], Iterator],
    async_fn: Callable[[AsyncIterable, *Ts], AsyncIterator],
    it: SupportsSlice | Iterable | AsyncIterable,
    *args: *Ts,
) -> Iterator | AsyncIterator:
    if isinstance(it, AsyncIterable):
        return async_fn(it, *args)

    if isinstance(it, Mapping) or not isinstance(it, SupportsSlice):
        return sync_fn(it, *args)

    if isinstance(it, str | bytes | tuple | list):  # Could always be sliced
        return slice_fn(it, *args)

    try:
        # Ensure that slice is supported by prefetching 1st item
        r = slice_fn(it, *args)
        first_or_none = tuple(islice(r, 1))
    except TypeError:
        return sync_fn(it, *args)  # type: ignore[arg-type]
    else:
        return chain(first_or_none, r)


# ----------------------------------------------------------------------------


def window_hint(it: Sized, size: int) -> int:
    return len(it) + 1 - size


def chunk_hint(it: Sized, size: int) -> int:
    return len(range(0, len(it), size))


def _sliced_windowed[T](s: SupportsSlice[T], size: int, /) -> Iterator[T]:
    len_ = len(s)
    if not len_:
        return iter([])
    if len_ < size:
        return iter([s[:]])
    indices = range(len_ + 1)
    slices = map(slice, indices[:-size], indices[size:])
    return map(s.__getitem__, slices)


def _windowed[T](it: Iterable[T], size: int, /) -> Iterator[tuple[T, ...]]:
    assert size >= 1

    it = iter(it)
    w = deque(islice(it, size), maxlen=size)

    if not w:
        return iter([])
    if len(w) < size:
        return iter([tuple(w)])
    return map(tuple, chain([w], map(w.__iadd__, zip(it))))


async def _awindowed[T](
    it: AsyncIterable[T], size: int, /
) -> AsyncGenerator[tuple[T, ...]]:
    assert size >= 1

    w = deque[T](maxlen=size)
    async for x in it:
        w.append(x)
        if len(w) == size:
            yield tuple(w)
    if w and len(w) < size:
        yield tuple(w)


def _sliced[T](s: SupportsSlice[T], size: int, /) -> Iterator[T]:
    assert size >= 1

    indices = range(len(s) + size)
    slices = map(slice, indices[::size], indices[size::size])
    return map(s.__getitem__, slices)


async def _abatched[T](
    it: AsyncIterable[T], size: int, /
) -> AsyncGenerator[tuple[T, ...]]:
    assert size >= 1

    batch: list[T] = []
    async for x in it:
        batch.append(x)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


# ---------------------------------------------------------------------------


@overload
def windowed[T](it: SupportsSlice[T], size: int, /) -> Iterator[T]: ...


@overload
def windowed[T](it: Iterable[T], size: int, /) -> Iterator[tuple[T, ...]]: ...


@overload
def windowed[T](
    __it: AsyncIterable[T], size: int, /
) -> AsyncIterator[tuple[T, ...]]: ...


def windowed(
    it: SupportsSlice | Iterable | AsyncIterable, size: int, /
) -> Iterator | AsyncIterator:
    """Retrieve overlapped windows from iterable.

    Tries to use slicing if possible. Supports async iterables.

    >>> [*windowed(range(6), 3)]
    [range(0, 3), range(1, 4), range(2, 5), range(3, 6)]

    >>> [*windowed(iter(range(6)), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    if size < 1:
        raise ValueError('size must be >= 1')
    return _dispatch(_sliced_windowed, _windowed, _awindowed, it, size)


@overload
def chunked[S](__it: SupportsSlice[S], size: int, /) -> Iterator[S]: ...


@overload
def chunked[T](__it: Iterable[T], size: int, /) -> Iterator[tuple[T, ...]]: ...


@overload
def chunked[T](
    __it: AsyncIterable[T], size: int, /
) -> AsyncIterator[tuple[T, ...]]: ...


def chunked(
    it: SupportsSlice | Iterable | AsyncIterable, size: int, /
) -> Iterator | AsyncIterator:
    """Split iterable to chunks of at most size items each.

    Uses slicing if possible.
    Each next() on result will advance passed iterable to size items.

    Supports async iterables.

    >>> [*chunked(range(10), 3)]
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]

    >>> [*chunked(iter(range(10)), 3)]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    if size < 1:
        raise ValueError('size must be >= 1')
    return _dispatch(_sliced, batched, _abatched, it, size)


# ----------------------------------------------------------------------------


def each_is[T](items: Sequence, tp: type[T]) -> TypeGuard[Sequence[T]]:
    return all(isinstance(it, tp) for it in items)


# ----------------------------------------------------------------------------


def unqueue[T](q: HasPopleft[T], /) -> Generator[T]:
    # Same as more_itertools.iter_except(q.popleft, IndexError)
    try:
        while True:
            yield q.popleft()
    except IndexError:
        return


# ----------------------------------------------------------------------------


def ichunked[T](it: Iterable[T], size: int, /) -> Generator[Iterator[T]]:
    """Split iterable to chunks of at most size items each.

    Does't consume items from passed iterable to return complete chunk
    unlike chunked, as yields iterators, not sequences.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> [[*chunk] for chunk in s]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    if size <= 0:
        raise ValueError('size must be >= 0')
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
        yield chain(unqueue(head), body, unqueue(tail))

        # Advance and fill internal cache, expand tail with items from body
        tail += body


# ----------------------------------------------------------------------------


def ilen(iterable: Iterable, /) -> int:
    """Return number of items in *iterable*.

    This consumes iterable, so handle with care.
    See `more_itertools.ilen`.
    """
    # zip(xs) -> (x0,), (x1,), ..., (xn,) - all are truthy
    # repeat(1) -> 1, 1, ...
    # compress(xs, ys) -> (x for x, y in zip(xs, ys) if y), and all Ys are true
    # thus its equal to sum(1 for _ in iterable)
    return sum(compress(repeat(1), zip(iterable)))


def eat(iterable: Iterable, /, *, daemon: bool = False) -> None:
    """
    Consume iterable, optionally in  background thread.

    See `more_itertools.consume`.
    """
    if daemon:
        Thread(target=deque, args=(iterable, 0), daemon=True).start()
    else:
        # feed the entire iterator into a zero-length deque
        deque(iterable, 0)


def roundrobin[T](*iterables: Iterable[T]) -> Generator[T]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    iters = cycle(iter(it) for it in iterables)
    for pending in range(len(iterables) - 1, -1, -1):
        yield from map(next, iters)
        iters = cycle(islice(iters, pending))


# ----------------------------------------------------------------------------


@overload
def groupby[T, K: Hashable](
    iterable: Iterable[T], /, key: Unary[T, K]
) -> dict[K, list[T]]: ...


@overload
def groupby[T, K: Hashable, V](
    iterable: Iterable[T], /, key: Unary[T, K], value: Unary[T, V]
) -> dict[K, list[V]]: ...


def groupby[T, K: Hashable](
    iterable: Iterable[T], /, key: Unary[T, K], value=lambda x: x
) -> dict[K, list]:
    """Group items from iterable by key.

    >>> groupby([True, (), 1, 0], bool)
    {True: [True, 1], False: [(), 0]}

    """
    r: dict[K, list] = {}
    for x in iterable:
        r.setdefault(key(x), []).append(value(x))
    return r


# ----------------------------------------------------------------------------


async def azip(*iterables: AnyIterable) -> AsyncGenerator[tuple]:
    if each_is(iterables, Iterable):  # type: ignore[type-abstract]
        for x in zip(*iterables):
            yield x
        return

    aiters = [
        _as_asyncgen(it) if isinstance(it, Iterable) else aiter(it)
        for it in iterables
    ]
    while True:
        tasks = [asyncio.ensure_future(ait.__anext__()) for ait in aiters]
        try:
            ret = await asyncio.gather(*tasks)
        except StopAsyncIteration:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        else:
            yield tuple(ret)


async def _as_asyncgen[T](it: Iterable[T]) -> AsyncGenerator[T]:
    for x in it:
        yield x
