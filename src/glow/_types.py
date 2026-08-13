import enum
from collections.abc import (
    AsyncIterable,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Sized,
)
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

type KeyFn[**P] = Callable[P, Hashable]

type Coro[T] = Coroutine[Any, Any, T]
type ACallable[**P, R] = Callable[P, Coro[R]]
type ASCallable[*Ts, R] = Callable[[*Ts], Coro[R]]
type AnyIterable[T] = AsyncIterable[T] | Iterable[T]

type Get[T] = Callable[[], T]
type Unary[T, R = object] = Callable[[T], R]
type AUnary[T, R = object] = ACallable[[T], R]

type CachePolicy = Literal['lru', 'mru']
type Maybe[T] = 'Some[T] | BaseException'


@dataclass(frozen=True, slots=True)
class Some[T]:
    x: T


class Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


class PsDecorator[**P](Protocol):
    def __call__[R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


class SupportsNext[T](Protocol):
    def __next__(self, /) -> T: ...


class SupportsSlice[T](Sized, Protocol):
    def __getitem__(self, s: slice, /) -> T: ...


class SupportsWrite[T = str](Protocol):
    def write(self, s: T, /) -> object: ...


class HasPopleft[T](Protocol):
    def popleft(self) -> T: ...


class Pipe[In, Out](Protocol):
    def send(self, value: In) -> Out: ...


class Empty(enum.Enum):
    token = 0


class QueueShutdownError(Exception):
    """Raised on access to terminated Queue."""


empty: Final = Empty.token
