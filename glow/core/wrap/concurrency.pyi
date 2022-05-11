from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import TypeVar, overload

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_BatchF = TypeVar('_BatchF', bound=Callable[[Iterable], list])


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    ...


def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]:
    ...


def call_once(fn: _F) -> _F:
    ...


def shared_call(fn: _F) -> _F:
    ...


@overload
def streaming(*,
              batch_size: int,
              timeouts: tuple[float, float] = ...,
              workers: int = ...) -> Callable[[_BatchF], _BatchF]:
    ...


@overload
def streaming(func: _BatchF,
              *,
              batch_size: int,
              timeouts: tuple[float, float] = ...,
              workers: int = ...) -> _BatchF:
    ...
