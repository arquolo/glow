from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

from ._types import Callback, Decorator, Get

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
def time_this[**P, R](
    fn: Callable[P, R], /, *, name: str | None = ..., disable: bool = ...
) -> Callable[P, R]: ...
@overload
def time_this(*, name: str | None = ..., disable: bool = ...) -> Decorator: ...
