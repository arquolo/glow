import asyncio
import concurrent.futures as cf
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Literal, Protocol, overload

type KeyFn[H: Hashable] = Callable[..., H]

type Coro[T] = Coroutine[Any, Any, T]
type AnyIterable[T] = AsyncIterable[T] | Iterable[T]
type AnyIterator[T] = AsyncIterator[T] | Iterator[T]

type BatchFn[T, R] = Callable[[Sequence[T]], Sequence[R]]
type ABatchFn[T, R] = Callable[[Sequence[T]], Coro[Sequence[R]]]

type AnyFuture[R] = cf.Future[R] | asyncio.Future[R]
type Job[T, R] = tuple[T, AnyFuture[R]]

type Get[T] = Callable[[], T]
type Callback[T] = Callable[[T], object]

type CachePolicy = Literal['lru', 'mru'] | None


@dataclass(frozen=True, slots=True)
class Some[T]:
    x: T


class Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


class BatchDecorator(Protocol):
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
class ABatchDecorator(Protocol):
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...


class AnyBatchDecorator(Protocol):
    @overload
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
    @overload
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...
