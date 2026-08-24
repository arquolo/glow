from typing import overload

from ._futures import (
    ABatchFn,
    ABatchFnRv,
    BatchDecorator,
    BatchFn,
    BatchFnRv,
    PsBatchDecorator,
    UsableSize,
)

@overload
def streaming(
    *,
    batch_size: int | UsableSize = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> BatchDecorator: ...
@overload
def streaming[T](
    *,
    batch_size: UsableSize[T],
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> PsBatchDecorator[T]: ...
@overload
def streaming[T, R](
    func: BatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> BatchFnRv[T, R]: ...
@overload
def streaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    pool_timeout: float | None = ...,
) -> ABatchFnRv[T, R]: ...
