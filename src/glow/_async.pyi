from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Required, Self, TypedDict, Unpack, overload

from ._futures import (
    ABatchDecorator,
    ABatchFn,
    ABatchFnRv,
    PsABatchDecorator,
    UsableSize,
)
from ._types import ACallable, AnyIterable, ASCallable

class _AmapKwargs(TypedDict, total=False):
    limit: Required[int]
    unordered: bool

def astarmap[*Ts, R](
    func: ASCallable[*Ts, R],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, R](
    func: ACallable[[T], R],
    iter1: AnyIterable[T],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, R](
    func: ACallable[[T, T2], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, T3, R](
    func: ACallable[[T, T2, T3], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, T3, T4, R](
    func: ACallable[[T, T2, T3, T4], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[R](
    func: ACallable[..., R],
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
async def amap_dict[K, T, T2](
    func: ACallable[[T], T2], obj: Mapping[K, T], /, *, limit: int
) -> dict[K, T2]: ...
@overload
def azip() -> AsyncGenerator[Any]: ...
@overload
def azip[T](iter1: AnyIterable[T], /) -> AsyncGenerator[tuple[T]]: ...  # noqa: RUF100,RUF102
@overload
def azip[T, T2](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], /
) -> AsyncGenerator[tuple[T, T2]]: ...
@overload
def azip[T, T2, T3](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], iter3: AnyIterable[T3], /
) -> AsyncGenerator[tuple[T, T2, T3]]: ...
@overload
def azip[T, T2, T3, T4](
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
) -> AsyncGenerator[tuple[T, T2, T3, T4]]: ...
@overload
def azip(
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
) -> AsyncGenerator[tuple]: ...
@overload
def astreaming(
    *, batch_size: int | UsableSize = ..., timeout: float = ...
) -> ABatchDecorator: ...
@overload
def astreaming[T](
    *, batch_size: UsableSize[T], timeout: float = ...
) -> PsABatchDecorator[T]: ...
@overload
def astreaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
) -> ABatchFnRv[T, R]: ...

class RwLock:
    def __init__(self) -> None: ...
    def read(self) -> AbstractAsyncContextManager: ...
    def write(self) -> AbstractAsyncContextManager: ...

class MulticastQueue[T]:
    def __init__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_) -> None: ...
    def __aiter__(self) -> AsyncGenerator[T]: ...
    async def get(self, idx: int) -> T: ...
    async def put(self, value: T) -> None: ...
    def subscribe(self) -> AsyncContextIterator[T]: ...
    def close(self) -> None: ...

class AsyncContextIterator[T]:
    def __enter__(self) -> Self: ...
    def __exit__(self, *_) -> None: ...
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> T: ...
    def close(self) -> None: ...
