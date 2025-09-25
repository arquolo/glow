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
from typing import Any, Literal, Protocol

type KeyFn[H: Hashable] = Callable[..., H]

type Coro[T] = Coroutine[Any, Any, T]
type AnyIterable[T] = AsyncIterable[T] | Iterable[T]
type AnyIterator[T] = AsyncIterator[T] | Iterator[T]

type Get[T] = Callable[[], T]
type Callback[T] = Callable[[T], object]

type CachePolicy = Literal['lru', 'mru'] | None


@dataclass(frozen=True, slots=True)
class Some[T]:
    x: T


class Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...
