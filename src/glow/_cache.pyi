from typing import Literal, overload

from ._futures import AnyBatchDecorator, PsAnyBatchDecorator
from ._types import CachePolicy, Decorator, KeyFn, PsDecorator

def cache_status() -> str: ...

# Unbound
@overload
def memoize() -> Decorator: ...
@overload
def memoize[**P](*, key_fn: KeyFn[P]) -> PsDecorator[P]: ...

# Unbound batched
@overload
def memoize(*, batched: Literal[True]) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    *, batched: Literal[True], key_fn: KeyFn[T]
) -> PsAnyBatchDecorator[T]: ...

# -------------------------------- time only ---------------------------------

# Time-capped
@overload
def memoize(*, ttl: float) -> Decorator: ...
@overload
def memoize[**P](*, key_fn: KeyFn[P], ttl: float) -> PsDecorator[P]: ...

# Time-capped batched
@overload
def memoize(*, batched: Literal[True], ttl: float) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    *, batched: Literal[True], key_fn: KeyFn[T], ttl: float
) -> PsAnyBatchDecorator[T]: ...

# ---------------------------------- count -----------------------------------

# Count/byte-capped
@overload
def memoize(
    count: int,
    *,
    nbytes: int | None = ...,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> Decorator: ...
@overload
def memoize[**P](
    count: int,
    *,
    nbytes: int | None = ...,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# Count/byte-capped batched
@overload
def memoize(
    count: int,
    *,
    nbytes: int | None = ...,
    batched: Literal[True],
    policy: CachePolicy = ...,
    ttl: float | None = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    count: int,
    *,
    nbytes: int | None = ...,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsAnyBatchDecorator[T]: ...

# ---------------------------------- bytes -----------------------------------

# Byte-capped
@overload
def memoize(
    *, nbytes: int, policy: CachePolicy = ..., ttl: float | None = ...
) -> Decorator: ...
@overload
def memoize[**P](
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    ttl: float | None = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsAnyBatchDecorator[T]: ...
