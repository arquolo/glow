from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Iterator,
)
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)

_T = TypeVar('_T')
_R = TypeVar('_R')
_P = ParamSpec('_P')

KeyFn: TypeAlias = Callable[_P, Hashable]

Coro: TypeAlias = Coroutine[Any, Any, _T]
AnyIterable: TypeAlias = AsyncIterable[_T] | Iterable[_T]
AnyIterator: TypeAlias = AsyncIterator[_T] | Iterator[_T]

Get: TypeAlias = Callable[[], _T]
Callback: TypeAlias = Callable[[_T], object]

CachePolicy: TypeAlias = Literal['lru', 'mru'] | None


@dataclass(frozen=True, slots=True)
class Some(Generic[_T]):
    x: _T


class Decorator(Protocol):
    def __call__(self, fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...


class PsDecorator(Protocol, Generic[_P]):
    def __call__(self, fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...
