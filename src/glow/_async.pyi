from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Required, TypedDict, Unpack, overload

from ._futures import ABatchDecorator, ABatchFn
from ._types import AnyIterable, Coro

class _AmapKwargs(TypedDict, total=False):
    limit: Required[int]
    unordered: bool

def astarmap[*Ts, R](
    func: Callable[[*Ts], Coro[R]],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[R]: ...
@overload
def amap[T, R](
    func: Callable[[T], Coro[R]],
    iter1: AnyIterable[T],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, R](
    func: Callable[[T, T2], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, R](
    func: Callable[[T, T2, T3], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[R]: ...
@overload
def amap[T, T2, T3, T4, R](
    func: Callable[[T, T2, T3, T4], Coro[R]],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
    **kwargs: Unpack[_AmapKwargs],
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
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[R]: ...
async def amap_dict[K, T1, T2](
    func: Callable[[T1], Coro[T2]], obj: Mapping[K, T1], /, *, limit: int
) -> dict[K, T2]: ...
@overload
def azip() -> AsyncIterator[Any]: ...
@overload
def azip[T](
    iter1: AnyIterable[T], /
) -> AsyncIterator[tuple[T]]: ...  # noqa: RUF100,Y090
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
@overload
def astreaming(
    *, batch_size: int | None = ..., timeout: float = ...
) -> ABatchDecorator: ...
@overload
def astreaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
) -> ABatchFn[T, R]: ...

class RwLock:
    def __init__(self) -> None: ...
    def read(self) -> AbstractAsyncContextManager: ...
    def write(self) -> AbstractAsyncContextManager: ...
