from typing import Literal, overload

from ._futures import AnyBatchDecorator, PsAnyBatchDecorator
from ._types import CachePolicy, Decorator, KeyFn, PsDecorator

def cache_status() -> str: ...

# ----------------------- non batched, non parametric ------------------------

# unbound or time-constrained
@overload
def memoize(*, ttl: float | None = ...) -> Decorator: ...

# byte-capped
@overload
def memoize(
    *, nbytes: int, policy: CachePolicy = ..., ttl: float | None = ...
) -> Decorator: ...

# count or optionally, byte-capped
@overload
def memoize(
    count: int,
    *,
    nbytes: int | None = ...,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> Decorator: ...

# ------------------------- non batched, parametric --------------------------

# unbound or time-constrained
@overload
def memoize[**P](
    *, key_fn: KeyFn[P], ttl: float | None = ...
) -> PsDecorator[P]: ...

# byte-capped
@overload
def memoize[**P](
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# count or optionally, byte-capped
@overload
def memoize[**P](
    count: int,
    *,
    nbytes: int | None = ...,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# ------------------------- batched, non parametric --------------------------

# unbound or time-constrained
@overload
def memoize(
    *, batched: Literal[True], ttl: float | None = ...
) -> AnyBatchDecorator: ...

# byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    ttl: float | None = ...,
) -> AnyBatchDecorator: ...

# count or optionally, byte-capped
@overload
def memoize(
    count: int,
    *,
    nbytes: int | None = ...,
    batched: Literal[True],
    policy: CachePolicy = ...,
    ttl: float | None = ...,
) -> AnyBatchDecorator: ...

# --------------------------- batched, parametric ----------------------------

# unbound or time-constrained
@overload
def memoize[T](
    *, batched: Literal[True], key_fn: KeyFn[T], ttl: float | None = ...
) -> PsAnyBatchDecorator[T]: ...

# byte-capped
@overload
def memoize[T](
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsAnyBatchDecorator[T]: ...

# count or optionally, byte-capped
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
