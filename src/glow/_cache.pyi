from typing import Literal, ParamSpec, TypeVar, overload

from ._futures import AnyBatchDecorator, PsAnyBatchDecorator
from ._types import CachePolicy, Decorator, KeyFn, PsDecorator

_T = TypeVar('_T')
_P = ParamSpec('_P')

def cache_status() -> str: ...

# Unbound
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
) -> Decorator: ...
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: KeyFn[_P],
) -> PsDecorator[_P]: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
) -> AnyBatchDecorator: ...
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: KeyFn[_T],
) -> PsAnyBatchDecorator[_T]: ...

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> Decorator: ...
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn[_P],
) -> PsDecorator[_P]: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[_T],
) -> PsAnyBatchDecorator[_T]: ...

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
) -> Decorator: ...
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn[_P],
) -> PsDecorator[_P]: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
) -> AnyBatchDecorator: ...
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn[_T],
) -> PsAnyBatchDecorator[_T]: ...
