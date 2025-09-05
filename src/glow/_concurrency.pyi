from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

from ._types import BatchFn

def threadlocal[T, **P](
    fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Callable[[], T]: ...
def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]: ...
def call_once[T](fn: Callable[[], T], /) -> Callable[[], T]: ...
def shared_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
def weak_memoize[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def streaming[F: BatchFn](
    *,
    batch_size: int,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> Callable[[F], F]: ...
@overload
def streaming[F: BatchFn](
    func: F,
    *,
    batch_size: int,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> F: ...
