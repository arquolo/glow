__all__ = ['init_loguru', 'span_task']

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypedDict, Unpack, overload

from loguru import FilterDict, FilterFunction, FormatFunction

from ._types import Coro

class _LoggerAddKwds(TypedDict, total=False):
    colorize: bool | None
    serialize: bool
    backtrace: bool
    diagnose: bool
    filter: str | FilterFunction | FilterDict

@overload
def init_loguru(
    level: str = ...,
    *,
    names: Iterable[str] | Mapping[str, Sequence[str]] = ...,
    fmt: str = ...,
    extra: bool = ...,
    **logger_add_kwargs: Unpack[_LoggerAddKwds],
) -> None: ...
@overload
def init_loguru(
    level: str = ...,
    *,
    names: Iterable[str] | Mapping[str, Sequence[str]] = ...,
    fmt: FormatFunction,
    **logger_add_kwargs: Unpack[_LoggerAddKwds],
) -> None: ...
@overload
def span_task() -> str | None: ...
@overload
def span_task(task_id: str, /) -> _TaskSpanner: ...

class _TaskSpanner:
    def __enter__(self) -> str: ...
    def __exit__(self, exc_type, exc, tb) -> bool | None: ...
    @overload
    def __call__[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](
        self, fn: Callable[P, Coro[R]]
    ) -> Callable[P, Coro[R]]: ...
