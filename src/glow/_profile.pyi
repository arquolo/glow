from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import ParamSpec, TypeVar, overload

from ._types import Callback, Decorator, Get

_R = TypeVar('_R')
_P = ParamSpec('_P')

def memprof(
    name_or_callback: str | Callback[float] | None = ..., /
) -> AbstractContextManager[None]: ...
@overload
def timer(
    name: str | None = ...,
    time: Get[int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def timer(
    callback: Callback[int] | None,
    time: Get[int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def time_this(
    fn: Callable[_P, _R], /, *, name: str | None = ..., disable: bool = ...
) -> Callable[_P, _R]: ...
@overload
def time_this(*, name: str | None = ..., disable: bool = ...) -> Decorator: ...
