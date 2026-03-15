__all__ = ['as_actor', 'coroutine']

from collections import Counter, deque
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator
from functools import update_wrapper
from threading import Lock
from typing import Generic, ParamSpec, TypeVar

import wrapt

from ._more import _deiter

_P = ParamSpec('_P')
_Y = TypeVar('_Y')
_S = TypeVar('_S')
_R = TypeVar('_R')
_T = TypeVar('_T')


def coroutine(
    fn: Callable[_P, Generator[_Y, _S, _R]], /
) -> Callable[_P, Generator[_Y, _S, _R]]:
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Generator[_Y, _S, _R]:
        coro = fn(*args, **kwargs)
        next(coro)
        return coro

    return update_wrapper(wrapper, fn)


class _Sync(wrapt.ObjectProxy, Generic[_Y, _S, _R]):  # type: ignore[misc]
    __wrapped__: Generator[_Y, _S, _R]

    def __init__(self, wrapped: Generator[_Y, _S, _R]) -> None:
        super().__init__(wrapped)
        self._self_lock = Lock()

    def _call(
        self, op: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        with self._self_lock:
            return op(*args, **kwargs)

    def __next__(self) -> _Y:
        return self._call(self.__wrapped__.__next__)

    def send(self, item: _S, /) -> _Y:
        return self._call(self.__wrapped__.send, item)

    def throw(self, value: BaseException, /) -> _Y:
        return self._call(self.__wrapped__.throw, value)

    def close(self) -> None:
        self._call(self.__wrapped__.close)


def threadsafe_iter(
    fn: Callable[_P, Generator[_Y, _S, _R]], /
) -> Callable[_P, Generator[_Y, _S, _R]]:
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Generator[_Y, _S, _R]:
        gen = fn(*args, **kwargs)
        return _Sync(gen)

    return update_wrapper(wrapper, fn)


@threadsafe_iter
@coroutine
def summary() -> Generator[None, Hashable | None]:
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
def as_actor(
    transform: Callable[[Iterable[_T]], Iterator[_R]],
) -> Generator[_R, _T]:
    buf = deque[_T]()
    gen = transform(_deiter(buf))  # infinite

    # shortcuts
    buf_append, gen_next = buf.append, gen.__next__

    x = yield  # type: ignore[misc]  # preseed coroutine
    while True:
        buf_append(x)
        x = yield gen_next()
