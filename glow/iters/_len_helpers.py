__all__ = ('as_sized', 'repeatable')

import functools
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar, overload

_T = TypeVar('_T')
_G = TypeVar('_G', bound=Callable)


@dataclass
class SizedIter(Iterator[_T]):
    it: Iterator[_T]
    size: int

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        if self.size:
            self.size -= 1
        return next(self.it)

    def __len__(self) -> int:
        return self.size


@overload
def as_sized(gen_fn: None = ..., *,
             hint: Callable[..., int]) -> Callable[[_G], _G]:
    ...


@overload
def as_sized(gen_fn: _G, *, hint: Callable[..., int]) -> _G:
    ...


def as_sized(gen_fn=None, *, hint):
    """Packs generator function with size hint, thus making it sized.

    `hint` - callable which returns `len` of result iterable.
    It should have same signature as `gen`.
    If it catches TypeError while calling it, skips packing.

    >>> @as_sized(hint=len)
    ... def make_squares(it):
    ...    return (x**2 for x in it)
    ...
    >>> squares = make_squares(range(5))
    >>> len(squares)
    5
    >>> list(squares)
    [0, 1, 4, 9, 16]
    >>> len(squares)
    0
    >>> squares = make_squares(x for x in range(5))
    >>> len(squares)
    Traceback (most recent call last):
        ...
    TypeError: object of type 'generator' has no len()
    >>> list(squares)
    [0, 1, 4, 9, 16]
    """
    if gen_fn is None:
        return functools.partial(as_sized, hint=hint)

    @functools.wraps(gen_fn)
    def wrapper(*args, **kwargs):
        gen = gen_fn(*args, **kwargs)
        try:
            return SizedIter(gen, hint(*args, **kwargs))
        except TypeError:
            return gen

    return wrapper


class _Repeatable(Iterable[_T]):
    def __init__(self,
                 hint: Callable[..., int],
                 gen_fn: Callable[..., Iterable[_T]],
                 *args, **kwargs) -> None:
        self.gen = functools.partial(gen_fn, *args, **kwargs)
        self.hint = hint

    def __iter__(self) -> Iterator[_T]:
        return iter(self.gen())

    def __len__(self) -> int:
        return self.hint(*self.gen.args, **self.gen.keywords)


@overload
def repeatable(gen_fn: None = ..., *,
               hint: Callable[..., int]) -> Callable[[_G], _G]:
    ...


@overload
def repeatable(gen_fn: _G, *, hint: Callable[..., int]) -> _G:
    ...


def repeatable(gen_fn=None, *, hint):
    if gen_fn is None:
        return functools.partial(repeatable, hint=hint)

    return functools.wraps(gen_fn)(
        functools.partial(_Repeatable, hint, gen_fn)
    )
