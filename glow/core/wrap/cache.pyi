from typing import Callable, Hashable, Literal, Sequence, TypeVar, overload

_F = TypeVar('_F', bound=Callable)
_Fbatch = TypeVar('_Fbatch', bound=Callable[[Sequence], list])
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]


@overload
def memoize(capacity: int,
            *,
            policy: _Policy = ...,
            key_fn: _KeyFn = ...) -> Callable[[_F], _F]:
    ...


@overload
def memoize(capacity: int,
            *,
            batched: Literal[True],
            policy: _Policy = ...,
            key_fn: _KeyFn = ...) -> Callable[[_Fbatch], _Fbatch]:
    ...
