import asyncio
import concurrent.futures as cf
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Iterator,
)
from dataclasses import dataclass
from typing import Any, Literal

type KeyFn = Callable[..., Hashable]

type Coro[T] = Coroutine[Any, Any, T]
type AnyIterable[T] = AsyncIterable[T] | Iterable[T]
type AnyIterator[T] = AsyncIterator[T] | Iterator[T]

type BatchFn[T, R] = Callable[[list[T]], Iterable[R]]
type ABatchFn[T, R] = Callable[[list[T]], Awaitable[Iterable[R]]]

type AnyFuture[R] = cf.Future[R] | asyncio.Future[R]

type Get[T] = Callable[[], T]
type Callback[T] = Callable[[T], object]

type CachePolicy = Literal['lru', 'mru'] | None


@dataclass(frozen=True, slots=True)
class Some[T]:
    x: T
