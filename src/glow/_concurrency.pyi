from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

from ._futures import BatchDecorator, BatchFn, PsBatchDecorator, UsableSize
from ._types import Get

def threadlocal[T, **P](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Get[T]: ...
def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]: ...
def call_once[T](fn: Get[T], /) -> Get[T]: ...
def shared_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
def weak_memoize[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
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
) -> BatchFn[T, R]: ...
