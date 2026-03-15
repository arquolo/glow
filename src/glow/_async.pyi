from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import (
    Any,
    Required,
    TypedDict,
    TypeVar,
    TypeVarTuple,
    Unpack,
    overload,
)

from ._futures import ABatchDecorator, ABatchFn
from ._types import AnyIterable, Coro

_K = TypeVar('_K')
_T = TypeVar('_T')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')
_T4 = TypeVar('_T4')
_R = TypeVar('_R')
_Ts = TypeVarTuple('_Ts')

class _AmapKwargs(TypedDict, total=False):
    limit: Required[int]
    unordered: bool

def astarmap(
    func: Callable[[*_Ts], Coro[_R]],
    iterable: AnyIterable[tuple[*_Ts]],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
@overload
def amap(
    func: Callable[[_T], Coro[_R]],
    iter1: AnyIterable[_T],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
@overload
def amap(
    func: Callable[[_T, _T2], Coro[_R]],
    iter1: AnyIterable[_T],
    iter2: AnyIterable[_T2],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
@overload
def amap(
    func: Callable[[_T, _T2, _T3], Coro[_R]],
    iter1: AnyIterable[_T],
    iter2: AnyIterable[_T2],
    iter3: AnyIterable[_T3],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
@overload
def amap(
    func: Callable[[_T, _T2, _T3, _T4], Coro[_R]],
    iter1: AnyIterable[_T],
    iter2: AnyIterable[_T2],
    iter3: AnyIterable[_T3],
    iter4: AnyIterable[_T4],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
@overload
def amap(
    func: Callable[..., Coro[_R]],
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncIterator[_R]: ...
async def amap_dict(
    func: Callable[[_T], Coro[_T2]], obj: Mapping[_K, _T], /, *, limit: int
) -> dict[_K, _T2]: ...
@overload
def azip() -> AsyncIterator[Any]: ...
@overload
def azip(
    iter1: AnyIterable[_T], /
) -> AsyncIterator[tuple[_T]]: ...  # noqa: RUF100,Y090
@overload
def azip(
    iter1: AnyIterable[_T], iter2: AnyIterable[_T2], /
) -> AsyncIterator[tuple[_T, _T2]]: ...
@overload
def azip(
    iter1: AnyIterable[_T], iter2: AnyIterable[_T2], iter3: AnyIterable[_T3], /
) -> AsyncIterator[tuple[_T, _T2, _T3]]: ...
@overload
def azip(
    iter1: AnyIterable[_T],
    iter2: AnyIterable[_T2],
    iter3: AnyIterable[_T3],
    iter4: AnyIterable[_T4],
    /,
) -> AsyncIterator[tuple[_T, _T2, _T3, _T4]]: ...
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
def astreaming(
    fn: ABatchFn[_T, _R],
    /,
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
) -> ABatchFn[_T, _R]: ...

class RwLock:
    def __init__(self) -> None: ...
    def read(self) -> AbstractAsyncContextManager: ...
    def write(self) -> AbstractAsyncContextManager: ...
