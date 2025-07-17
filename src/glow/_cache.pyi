from collections.abc import Callable, Hashable, Iterable
from typing import Literal, Protocol, overload

type _Policy = Literal['lru', 'mru'] | None
type _KeyFn = Callable[..., Hashable]

def cache_status() -> str: ...

class _BatchedFn[T, R]:
    def __call__(self, _args: list[T], /) -> Iterable[R]: ...

class _AsyncBatchedFn[T, R]:
    async def __call__(self, _args: list[T], /) -> Iterable[R]: ...

class _Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...

class _BatchedDecorator(Protocol):
    @overload
    def __call__[T, R](
        self,
        fn: _BatchedFn[T, R],
        /,
    ) -> _BatchedFn[T, R]: ...
    @overload
    def __call__[T, R](
        self,
        fn: _AsyncBatchedFn[T, R],
        /,
    ) -> _AsyncBatchedFn[T, R]: ...

# Unbound
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: _KeyFn = ...,
) -> _Decorator: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: _KeyFn = ...,
) -> _BatchedDecorator: ...

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
) -> _Decorator: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
) -> _BatchedDecorator: ...

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
) -> _Decorator: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
) -> _BatchedDecorator: ...
