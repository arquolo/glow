__all__ = ('make_sized', )

from contextlib import suppress
from dataclasses import dataclass
from functools import partial, wraps
from typing import Callable, Iterable, Iterator, Sized, TypeVar, Union

T = TypeVar('T')


@dataclass
class SizedIter(Sized, Iterable[T]):
    """Wrapper for iterators/generator functions, packs them with `length`"""
    iter_: Union[Iterable[T], Callable[[], Iterable[T]]]
    length: int = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[T]:
        return iter(self.iter_() if callable(self.iter_) else self.iter_)


def default_hint(*args, **_):
    for iterable in args:
        with suppress(TypeError):
            return len(iterable)
    return None


def make_sized(gen=None, *, hint: Callable[..., int] = default_hint):
    """Packs generator function with size hint, thus making it sized.

    `hint` - callable which returns size of result iterable.
    It should have same signature as `gen`.

    >>> @make_sized(hint=lambda n: n)
    ... def gen_fn(n):
    ...    return (x for x in range(n))
    ...
    >>> gen = gen_fn(5)
    >>> list(gen)
    [0, 1, 2, 3, 4]
    >>> list(gen)
    [0, 1, 2, 3, 4]
    >>> len(gen)
    5
    """
    if gen is None:
        return partial(make_sized, hint=hint)

    @wraps(gen)
    def wrapper(*args, **kwargs):
        if hint is not None:
            try:
                length = hint(*args, **kwargs)
            except TypeError:
                length = None

        return SizedIter(partial(gen, *args, **kwargs), length=length)

    return wrapper
