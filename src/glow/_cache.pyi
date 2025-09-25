from typing import Literal, overload

from ._futures import AnyBatchDecorator
from ._types import CachePolicy, Decorator, KeyFn

def cache_status() -> str: ...

# Unbound
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    key_fn: KeyFn = ...,
) -> Decorator: ...

# Unbound batched
@overload
def memoize(
    count: None = ...,
    *,
    policy: None = ...,
    batched: Literal[True],
    key_fn: KeyFn = ...,
) -> AnyBatchDecorator: ...

# Count-capped
@overload
def memoize(
    count: int,
    *,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> Decorator: ...

# Count-capped batched
@overload
def memoize(
    count: int,
    *,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> AnyBatchDecorator: ...

# Byte-capped
@overload
def memoize(
    *,
    nbytes: int,
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> Decorator: ...

# Byte-capped batched
@overload
def memoize(
    *,
    nbytes: int,
    batched: Literal[True],
    policy: CachePolicy = ...,
    key_fn: KeyFn = ...,
) -> AnyBatchDecorator: ...
