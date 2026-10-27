__all__ = ['remote']

from collections.abc import Callable
from typing import overload

from ._types import Decorator

@overload
def remote(
    *,
    num_workers: int | None = ...,
    chunk_size: int = ...,
    latency: float = ...,
    prefetch: int = ...,
) -> Decorator: ...
@overload
def remote[**P, R](
    fn: Callable[P, R],
    *,
    num_workers: int | None = ...,
    chunk_size: int = ...,
    latency: float = ...,
    prefetch: int = ...,
) -> Callable[P, R]: ...
