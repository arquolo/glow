from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar, overload

_F = TypeVar('_F', bound=Callable)


def memprof(name_or_callback: str | Callable[[float], object] | None = ...,
            /) -> AbstractContextManager[None]:
    ...


@overload
def timer(name: str | None = ...,
          time: Callable[[], int] = ...,
          /,
          *,
          disable: bool = ...) -> AbstractContextManager[None]:
    ...


@overload
def timer(callback: Callable[[int], object] | None,
          time: Callable[[], int] = ...,
          /,
          *,
          disable: bool = ...) -> AbstractContextManager[None]:
    ...


@overload
def time_this(fn: _F, /, *, name: str | None = ..., disable: bool = ...) -> _F:
    ...


@overload
def time_this(*,
              name: str | None = ...,
              disable: bool = ...) -> Callable[[_F], _F]:
    ...
