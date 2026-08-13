from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

from ._types import Decorator, Get, Unary

def memprof(
    name_or_callback: str | Unary[float] | None = ..., /
) -> AbstractContextManager[None]: ...
def memtrack(
    callback: Unary[int, None] = ..., period: float = ...
) -> None: ...
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
    callback: Unary[int] | None,
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
def whereami(skip: int = 0, limit: int | None = None) -> str: ...
