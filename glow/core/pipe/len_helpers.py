__all__ = ('as_sized', 'repeatable')

import functools
from itertools import islice
from typing import Callable, Iterable, Iterator, TypeVar, overload

from typing_extensions import Protocol

from .._patch_len import len_hint

_T_co = TypeVar('_T_co', covariant=True)
_G = TypeVar('_G', bound=Callable)


class SizedIterable(Protocol[_T_co]):
    def __iter__(self) -> Iterator[_T_co]:
        ...

    def __len__(self) -> int:
        ...


class SizedIter(islice):  # type: ignore
    pass


@len_hint.register(SizedIter)
def _len_islice(x):
    _, (src, start, *stop_step), done = x.__reduce__()
    if not stop_step:
        return 0
    try:
        total = len(src) + done
    except TypeError:
        total = float('+Inf')
    stop, step = stop_step
    stop = total if stop is None else min(total, stop)
    return len(range(start, stop, step))


_SizeHint = Callable[..., int]
_GenFn = Callable[..., Iterable[_T_co]]
_SizedGenFn = Callable[..., SizedIterable[_T_co]]

# ---------------------------------------------------------------------------


@overload
def as_sized(
    hint: _SizeHint
) -> Callable[[Callable[..., Iterable[_T_co]]], _SizedGenFn[_T_co]]:
    ...


@overload
def as_sized(gen_fn: Callable[..., Iterable[_T_co]],
             hint: _SizeHint) -> Callable[..., SizedIterable[_T_co]]:
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
    >>> [*squares]
    [0, 1, 4, 9, 16]
    >>> len(squares)
    0
    >>> squares = make_squares(x for x in range(5))
    >>> len(squares)
    Traceback (most recent call last):
        ...
    TypeError: object of type 'generator' has no len()
    >>> [*squares]
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


# ---------------------------------------------------------------------------


class _Repeatable(Iterable[_T_co]):
    def __init__(self, hint: Callable[..., int],
                 gen_fn: Callable[..., Iterable[_T_co]], *args: object,
                 **kwargs: object) -> None:
        self.gen = functools.partial(gen_fn, *args, **kwargs)
        self.hint = hint

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self.gen())

    def __len__(self) -> int:
        return self.hint(*self.gen.args, **self.gen.keywords)


@overload
def repeatable(
    hint: _SizeHint
) -> Callable[[Callable[..., Iterable[_T_co]]], _SizedGenFn[_T_co]]:
    ...


@overload
def repeatable(gen_fn: Callable[..., Iterable[_T_co]],
               hint: _SizeHint) -> Callable[..., SizedIterable[_T_co]]:
    ...


def repeatable(gen_fn=None, *, hint):
    if gen_fn is None:
        return functools.partial(repeatable, hint=hint)

    return functools.wraps(gen_fn)(
        functools.partial(_Repeatable, hint, gen_fn))
