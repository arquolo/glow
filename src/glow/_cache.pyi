from collections.abc import Callable
from typing import Literal, SupportsInt, overload

from ._futures import BatchDecorator, PsBatchDecorator
from ._types import CachePolicy, Decorator, Get, KeyFn, PsDecorator

def cache_status() -> str: ...
def call_once[T](fn: Get[T], /) -> Get[T]: ...
def coalesce[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...

# ----------------------- non batched, non parametric ------------------------

# unbound or time-constrained
@overload
def memoize(
    *,
    batched: Literal[False] = ...,
    ttl: float | None = ...,
) -> Decorator: ...

# byte-capped
@overload
def memoize(
    *,
    nbytes: SupportsInt,
    batched: Literal[False] = ...,
    policy: CachePolicy | None = ...,
    ttl: float | None = ...,
) -> Decorator: ...

# count or optionally, byte-capped
@overload
def memoize(
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[False] = ...,
    policy: CachePolicy | None = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> Decorator: ...

# ------------------------- non batched, parametric --------------------------

# unbound or time-constrained
@overload
def memoize[**P](
    *,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# byte-capped
@overload
def memoize[**P](
    *,
    nbytes: SupportsInt,
    policy: CachePolicy | None = ...,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# count or optionally, byte-capped
@overload
def memoize[**P](
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    policy: CachePolicy | None = ...,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> PsDecorator[P]: ...

# ------------------------- batched, non parametric --------------------------

# unbound or time-constrained
@overload
def memoize(
    *, batched: Literal[True], ttl: float | None = ...
) -> BatchDecorator: ...

# byte-capped
@overload
def memoize(
    *,
    nbytes: SupportsInt,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    ttl: float | None = ...,
) -> BatchDecorator: ...

# count or optionally, byte-capped
@overload
def memoize(
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    ttl: float | None = ...,
) -> BatchDecorator: ...

# --------------------------- batched, parametric ----------------------------

# unbound or time-constrained
@overload
def memoize[T](
    *, batched: Literal[True], key_fn: KeyFn[T], ttl: float | None = ...
) -> PsBatchDecorator[T]: ...

# byte-capped
@overload
def memoize[T](
    *,
    nbytes: SupportsInt,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsBatchDecorator[T]: ...

# count or optionally, byte-capped
@overload
def memoize[T](
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> PsBatchDecorator[T]: ...
