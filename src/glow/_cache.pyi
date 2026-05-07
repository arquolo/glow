from typing import Literal, overload

from ._futures import AnyBatchDecorator, PsAnyBatchDecorator
from ._types import CachePolicy, Decorator, KeyFn, PsDecorator

def cache_status() -> str: ...

# Unbound
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    ttl: None = ...,
) -> Decorator: ...
@overload
def memoize[**P](
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: KeyFn[P],
    ttl: None = ...,
) -> PsDecorator[P]: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    ttl: None = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: KeyFn[T],
    ttl: None = ...,
) -> PsAnyBatchDecorator[T]: ...

# ---------------------------------- count -----------------------------------

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> Decorator: ...
@overload
def memoize[**P](
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    ttl: float | None = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsAnyBatchDecorator[T]: ...

# ---------------------------------- bytes -----------------------------------

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    ttl: float | None = ...,
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
