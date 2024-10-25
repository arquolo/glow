from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

def memprof(
    name_or_callback: str | Callable[[float], object] | None = ..., /
) -> AbstractContextManager[None]: ...
@overload
def timer(
    name: str | None = ...,
    time: Callable[[], int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def timer(
    callback: Callable[[int], object] | None,
    time: Callable[[], int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def time_this[
    **P, R
](
    fn: Callable[P, R], /, *, name: str | None = ..., disable: bool = ...
) -> Callable[P, R]: ...
@overload
def time_this[
    **P, R
](*, name: str | None = ..., disable: bool = ...) -> Callable[
    [Callable[P, R]], Callable[P, R]
]: ...
