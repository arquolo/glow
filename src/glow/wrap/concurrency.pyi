from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import overload

type _BatchedFn = Callable[[list], Iterable]

def threadlocal[
    T, **P
](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Callable[
    [], T
]: ...
def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]: ...
def call_once[F: Callable](fn: F, /) -> F: ...
def shared_call[F: Callable](fn: F, /) -> F: ...
def weak_memoize[F: Callable](fn: F, /) -> F: ...
@overload
def streaming[
    F: _BatchedFn
](
    *,
    batch_size: int,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> Callable[[F], F]: ...
@overload
def streaming[
    F: _BatchedFn
](
    func: F,
    *,
    batch_size: int,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> F: ...
