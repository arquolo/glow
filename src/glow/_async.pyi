from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
)
from typing import Any, overload

type _AnyIterable[T] = Iterable[T] | AsyncIterable[T]
type _Coro[T] = Coroutine[Any, Any, T]

def astarmap[*Ts, R](
    func: Callable[[*Ts], _Coro[R]],
    iterable: _AnyIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, R](
    func: Callable[[T], _Coro[R]],
    iter1: _AnyIterable[T],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, R](
    func: Callable[[T, T2], _Coro[R]],
    iter1: _AnyIterable[T],
    iter2: _AnyIterable[T2],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, R](
    func: Callable[[T, T2, T3], _Coro[R]],
    iter1: _AnyIterable[T],
    iter2: _AnyIterable[T2],
    iter3: _AnyIterable[T3],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, T4, R](
    func: Callable[[T, T2, T3, T4], _Coro[R]],
    iter1: _AnyIterable[T],
    iter2: _AnyIterable[T2],
    iter3: _AnyIterable[T3],
    iter4: _AnyIterable[T4],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[R](
    func: Callable[..., _Coro[R]],
    iter1: _AnyIterable,
    iter2: _AnyIterable,
    iter3: _AnyIterable,
    iter4: _AnyIterable,
    iter5: _AnyIterable,
    /,
    *iters: _AnyIterable,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def azip() -> AsyncIterator[Any]: ...
@overload
def azip[T](iter1: _AnyIterable[T], /) -> AsyncIterator[tuple[T]]: ...
@overload
def azip[T, T2](
    iter1: _AnyIterable[T], iter2: _AnyIterable[T2], /
) -> AsyncIterator[tuple[T, T2]]: ...
@overload
def azip[T, T2, T3](
    iter1: _AnyIterable[T], iter2: _AnyIterable[T2], iter3: _AnyIterable[T3], /
) -> AsyncIterator[tuple[T, T2, T3]]: ...
@overload
def azip[T, T2, T3, T4](
    iter1: _AnyIterable[T],
    iter2: _AnyIterable[T2],
    iter3: _AnyIterable[T3],
    iter4: _AnyIterable[T4],
    /,
) -> AsyncIterator[tuple[T, T2, T3, T4]]: ...
@overload
def azip(
    iter1: _AnyIterable,
    iter2: _AnyIterable,
    iter3: _AnyIterable,
    iter4: _AnyIterable,
    iter5: _AnyIterable,
    /,
    *iters: _AnyIterable,
) -> AsyncIterator[tuple]: ...
