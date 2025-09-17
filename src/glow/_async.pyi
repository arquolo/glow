from collections.abc import AsyncIterator, Callable
from typing import Any, overload

from ._types import ABatchFn, AnyIterable, Coro

def astarmap[*Ts, R](
    func: Callable[[*Ts], Coro[R]],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, R](
    func: Callable[[T], Coro[R]],
    iter1: AnyIterable[T],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, R](
    func: Callable[[T, T2], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, R](
    func: Callable[[T, T2, T3], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, T4, R](
    func: Callable[[T, T2, T3, T4], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
    *,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def amap[R](
    func: Callable[..., Coro[R]],
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
    limit: int,
    unordered: bool = ...,
) -> AsyncIterator[R]: ...
@overload
def azip() -> AsyncIterator[Any]: ...
@overload
def azip[T](iter1: AnyIterable[T], /) -> AsyncIterator[tuple[T]]: ...
@overload
def azip[T, T2](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], /
) -> AsyncIterator[tuple[T, T2]]: ...
@overload
def azip[T, T2, T3](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], iter3: AnyIterable[T3], /
) -> AsyncIterator[tuple[T, T2, T3]]: ...
@overload
def azip[T, T2, T3, T4](
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
) -> AsyncIterator[tuple[T, T2, T3, T4]]: ...
@overload
def azip(
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
) -> AsyncIterator[tuple]: ...
def astreaming[F: ABatchFn](
    batch_size: int = ..., timeout: float = ...
) -> Callable[[F], F]: ...
