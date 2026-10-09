__all__ = ['consumer']

from collections import Counter
from collections.abc import Callable, Generator, Hashable
from contextlib import AbstractContextManager
from functools import update_wrapper
from threading import Lock

try:
    from wrapt import BaseObjectProxy as ObjectProxy  # wrapt>=2.0
except ImportError:
    from wrapt import ObjectProxy

from ._types import SupportsNext


def consumer[**P, R: SupportsNext](fn: Callable[P, R], /) -> Callable[P, R]:
    """See `more_itertools.consumer`"""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        gen = fn(*args, **kwargs)
        next(gen)
        return gen

    return update_wrapper(wrapper, fn)


class _SyncGenerator[Y, S, R](ObjectProxy):
    __wrapped__: Generator[Y, S, R]

    def __init__(self, wrapped: Generator[Y, S, R]) -> None:
        super().__init__(wrapped)
        self._self_lock = Lock()

    def __iter__(self) -> Generator[Y, S, R]:
        return self

    def __next__(self) -> Y:
        return call_with(self._self_lock, self.__wrapped__.__next__)

    def send(self, item: S, /) -> Y:
        return call_with(self._self_lock, self.__wrapped__.send, item)

    def throw(self, *args) -> Y:
        return call_with(self._self_lock, self.__wrapped__.throw, *args)

    def close(self) -> R | None:
        return call_with(self._self_lock, self.__wrapped__.close)


def call_with[**P, R](
    cm: AbstractContextManager,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    with cm:
        return fn(*args, **kwargs)


def threadsafe_iter[**P, Y, S, R](
    fn: Callable[P, Generator[Y, S, R]], /
) -> Callable[P, Generator[Y, S, R]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Y, S, R]:
        gen = fn(*args, **kwargs)
        return _SyncGenerator(gen)

    return update_wrapper(wrapper, fn)


@threadsafe_iter
@consumer
def summary() -> Generator[dict[Hashable, int], Hashable | None]:
    # ? delete this or find use case
    state = Counter[Hashable]()
    while True:
        key = yield dict(state)
        if key is None:
            state.clear()
        else:
            state[key] += 1
            print(dict(state), flush=True, end='\r')
