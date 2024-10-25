from collections.abc import Callable, Hashable, Iterable
from typing import Literal, overload

type _BatchedFn = Callable[[list], Iterable]
type _Policy = Literal['raw', 'lru', 'mru']
type _KeyFn = Callable[..., Hashable]

@overload
def memoize[
    F: Callable
](
    capacity: int,
    *,
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
    bytesize: bool = ...,
) -> Callable[[F], F]: ...
@overload
def memoize[
    F: _BatchedFn
](
    capacity: int,
    *,
    batched: Literal[True],
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
    bytesize: bool = ...,
) -> Callable[[F], F]: ...
