from collections.abc import Callable
from typing import Literal, Protocol, overload

from ._types import ABatchFn, BatchFn, CachePolicy, KeyFn

def cache_status() -> str: ...

class _Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...

class _BatchDecorator(Protocol):
    @overload
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
    @overload
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...

# Unbound
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: KeyFn = ...,
) -> _Decorator: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: KeyFn = ...,
) -> _BatchDecorator: ...

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> _Decorator: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> _BatchDecorator: ...

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> _Decorator: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> _BatchDecorator: ...
