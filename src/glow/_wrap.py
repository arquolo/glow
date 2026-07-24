__all__ = ['wrap']

import types
from collections.abc import Callable, Generator, Iterator
from functools import partial
from typing import Protocol, Self

try:
    from wrapt import BaseObjectProxy as ObjectProxy  # wrapt>=2.0
except ImportError:
    from wrapt import ObjectProxy

from ._dev import hide_frame
from ._types import Get

_OP_FORK_STOPITER = True
_OP_FUNC = True


def wrap[**P, R](func: Callable[P, R], wrapper: 'Wrapper') -> Callable[P, R]:
    return _Callable(func, wrapper)


class Wrapper(Protocol):
    def new_call(self) -> None: ...

    # This one start right after function was called,
    # and stops right before it's called again.
    # Usage:
    #   fn(*args, **kwargs)
    #   resume = wrapper.suspend()
    #   ...
    #   resume()
    #   fn(*args, **kwargs)
    def suspend(self) -> Get[None]: ...

    # This one start right before function was called,
    # and stops right after it returned.
    # Usage:
    #   return wrapper(fn, *args, **kwargs)
    def __call__[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwds: P.kwargs
    ) -> R: ...


class _Proxy[T](ObjectProxy):
    __wrapped__: T

    def __init__(self, wrapped: T, wrapper: Wrapper) -> None:
        super().__init__(wrapped)
        self._self_wrapper = wrapper


class _Callable[**P, R](_Proxy[Callable[P, R]]):
    def __get__(
        self, instance: object, owner: type | None
    ) -> '_BoundCallable':
        fn = self.__wrapped__.__get__(instance, owner)
        return _BoundCallable(fn, self._self_wrapper)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # patch & record fn.__call__
        self._self_wrapper.new_call()
        with hide_frame:
            r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        return _wrap(r, self._self_wrapper)


class _BoundCallable[**P, R](_Callable[P, R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


# ---------------------------------- bases -----------------------------------


class _Iterator[Y](_Proxy[Iterator[Y]]):
    def __iter__(self) -> Iterator[Y]:
        itr = self._self_wrapper(self.__wrapped__.__iter__)
        if itr is self.__wrapped__:
            return self
        return _wrap(itr, self._self_wrapper)

    def __next__(self) -> Y:
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.__next__)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        return _wrap(ret, self._self_wrapper)


class _CoroLike[Y, S, R](_Proxy[Generator[Y, S, R]]):
    def send(self, value: S, /) -> Y:
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.send, value)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        return _wrap(ret, self._self_wrapper)

    def throw(self, *args) -> Y:
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.throw, *args)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        return _wrap(ret, self._self_wrapper)

    def close(self) -> R | None:
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.close)


class _Generator[Y, S, R](_CoroLike[Y, S, R], _Iterator[Y]):
    pass


# ------------------------------ *type wrappers ------------------------------


def _gen[Y, S, R](
    gen: types.GeneratorType[Y, S, R], wrapper: Wrapper
) -> Generator[Y, S, R]:
    assert iter(gen) is gen
    op: Get[Y] = gen.__next__
    try:
        while True:
            with hide_frame:
                item = wrapper(op)

            try:
                with hide_frame:
                    send = yield _wrap(item, wrapper)
            except BaseException as exc:  # noqa: BLE001
                op = partial(gen.throw, exc)
            else:
                op = gen.__next__ if send is None else partial(gen.send, send)

    except StopIteration as e:
        return _wrap(e, wrapper).value


# -------------------------------- decoration --------------------------------


def _wrap[T](r: T, wrapper: Wrapper) -> T:
    # function, generator, coroutine & async generator
    # are distinguishable only by their result
    if isinstance(r, _Proxy):
        return r

    if isinstance(r, StopIteration) and _OP_FORK_STOPITER:
        r.value = _wrap(r.value, wrapper)
        return r

    match r:
        # __iter__, __next__, send, throw, close
        case types.GeneratorType() if _OP_FUNC:  # genfuncs
            return _gen(r, wrapper)  # type: ignore[return-value]
        case Generator():  # user's generators
            return _Generator(r, wrapper)  # type: ignore[return-value]

        # __iter__, __next__
        case Iterator():
            return _Iterator(r, wrapper)  # type: ignore[return-value]

    return r
