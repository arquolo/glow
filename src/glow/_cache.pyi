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
) -> Decorator: ...
@overload
def memoize[**P](
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: KeyFn[P],
) -> PsDecorator[P]: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: KeyFn[T],
) -> PsAnyBatchDecorator[T]: ...

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> Decorator: ...
@overload
def memoize[**P](
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
) -> PsDecorator[P]: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
) -> PsAnyBatchDecorator[T]: ...

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
) -> Decorator: ...
@overload
def memoize[**P](
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn[P],
) -> PsDecorator[P]: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize[T](
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[T],
) -> PsAnyBatchDecorator[T]: ...
