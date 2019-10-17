__all__ = ('make_sized', )

from contextlib import suppress
from dataclasses import dataclass
from functools import partial, wraps
from typing import Callable, Iterable, Iterator, Sized, TypeVar

T = TypeVar('T')


@dataclass
class _SizedGenerator(Sized, Iterable[T]):
    """Wrapper for generator functions, packs them with `length`"""
    iterator_fn: Callable[..., Iterable[T]]
    length: int = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[T]:
        yield from self.iterator_fn()


def default_hint(*args, **_):
    for iterable in args:
        with suppress(TypeError):
            return len(iterable)
    return None


def make_sized(gen=None, *, hint=default_hint):
    """Packs generator function with size hint, thus making it sized.

    If `hint` is passed, it should have same signature as `gen`.

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

        return _SizedGenerator(partial(gen, *args, **kwargs), length=length)

    return wrapper
