from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import ParamSpec, TypeVar, overload

from ._futures import BatchDecorator, BatchFn
from ._types import Get

_T = TypeVar('_T')
_R = TypeVar('_R')
_P = ParamSpec('_P')

def threadlocal(
    fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
) -> Get[_T]: ...
def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]: ...
def call_once(fn: Get[_T], /) -> Get[_T]: ...
def shared_call(fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...
def weak_memoize(fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...
@overload
def streaming(
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchDecorator: ...
@overload
def streaming(
    func: BatchFn[_T, _R],
    /,
    *,
    batch_size: int | None = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float = ...,
) -> BatchFn[_T, _R]: ...
