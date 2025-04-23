from collections.abc import Callable, Hashable, Iterable
from typing import Literal, overload

type _BatchedFn[T, R] = Callable[[list[T]], Iterable[R]]
type _Policy = Literal['raw', 'lru', 'mru']
type _KeyFn = Callable[..., Hashable]

@overload
def memoize[**P, R](
    capacity: int,
    *,
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
    bytesize: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def memoize[T, R](
    capacity: int,
    *,
    batched: Literal[True],
    policy: _Policy = ...,
    key_fn: _KeyFn = ...,
    bytesize: bool = ...,
) -> Callable[[_BatchedFn[T, R]], _BatchedFn[T, R]]: ...
