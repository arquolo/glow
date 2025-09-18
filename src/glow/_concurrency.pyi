from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

from ._types import BatchDecorator, BatchFn

def threadlocal[T, **P](
    fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Callable[[], T]: ...
def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]: ...
def call_once[T](fn: Callable[[], T], /) -> Callable[[], T]: ...
def shared_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
def weak_memoize[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def streaming(
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchDecorator: ...
@overload
def streaming[T, R](
    func: BatchFn[T, R],
    /,
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchFn[T, R]: ...
