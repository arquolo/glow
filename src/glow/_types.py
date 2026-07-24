import enum
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Iterator,
    Sized,
)
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

type KeyFn[**P] = Callable[P, Hashable]

type Coro[T] = Coroutine[Any, Any, T]
type AnyIterable[T] = AsyncIterable[T] | Iterable[T]
type AnyIterator[T] = AsyncIterator[T] | Iterator[T]

type Get[T] = Callable[[], T]
type Callback[T] = Callable[[T], object]

type CachePolicy = Literal['lru', 'mru'] | None
type Maybe[T] = 'Some[T] | BaseException'


@dataclass(frozen=True, slots=True)
class Some[T]:
    x: T


class Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


class PsDecorator[**P](Protocol):
    def __call__[R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


class SupportsSlice[T](Sized, Protocol):
    def __getitem__(self, s: slice, /) -> T: ...


class SupportsWrite(Protocol):
    def write(self, s: str, /) -> object: ...


class HasPopleft[T](Protocol):
    def popleft(self) -> T: ...


class Empty(enum.Enum):
    token = 0


empty: Final = Empty.token
