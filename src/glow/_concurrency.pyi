from collections.abc import Callable
from typing import overload

from ._futures import (
    BatchDecorator,
    BatchFn,
    BatchFnRv,
    PsBatchDecorator,
    UsableSize,
)
from ._types import Get

def threadlocal[T, **P](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Get[T]: ...
@overload
def streaming(
    *,
    batch_size: int | UsableSize = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchDecorator: ...
@overload
def streaming[T](
    *,
    batch_size: UsableSize[T],
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> PsBatchDecorator[T]: ...
@overload
def streaming[T, R](
    func: BatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchFnRv[T, R]: ...
