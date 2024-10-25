__all__ = ['as_actor', 'coroutine']

from collections import Counter, deque
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator
from functools import update_wrapper
from threading import Lock

import wrapt

from ._more import _deiter


def coroutine[
    **P, Y, S, R
](fn: Callable[P, Generator[Y, S, R]], /) -> Callable[P, Generator[Y, S, R]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Y, S, R]:
        coro = fn(*args, **kwargs)
        next(coro)
        return coro

    return update_wrapper(wrapper, fn)


class _Sync[Y, S, R](wrapt.ObjectProxy):
    __wrapped__: Generator[Y, S, R]

    def __init__(self, wrapped: Generator[Y, S, R]) -> None:
        super().__init__(wrapped)
        self._self_lock = Lock()

    def _call[
        **P, T
    ](self, op: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        with self._self_lock:
            return op(*args, **kwargs)

    def __next__(self) -> Y:
        return self._call(self.__wrapped__.__next__)

    def send(self, item: S, /) -> Y:
        return self._call(self.__wrapped__.send, item)

    def throw(self, value: BaseException, /) -> Y:
        return self._call(self.__wrapped__.throw, value)

    def close(self) -> None:
        return self._call(self.__wrapped__.close)


def threadsafe_iter[
    **P, Y, S, R
](fn: Callable[P, Generator[Y, S, R]], /) -> Callable[P, Generator[Y, S, R]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Y, S, R]:
        gen = fn(*args, **kwargs)
        return _Sync(gen)

    return update_wrapper(wrapper, fn)


@threadsafe_iter
@coroutine
def summary() -> Generator[None, Hashable | None, None]:
    # ? delete this or find use case
    state = Counter[Hashable]()
    while True:
        key = yield
        if key is None:
            state.clear()
        else:
            state[key] += 1
            print(dict(state), flush=True, end='\r')


@threadsafe_iter
@coroutine
def as_actor[
    T, R
](transform: Callable[[Iterable[T]], Iterator[R]]) -> Generator[R, T, None]:
    buf = deque[T]()
    gen = transform(_deiter(buf))  # infinite

    # shortcuts
    buf_append, gen_next = buf.append, gen.__next__

    x = yield  # type: ignore[misc]  # preseed coroutine
    while True:
        buf_append(x)
        x = yield gen_next()
