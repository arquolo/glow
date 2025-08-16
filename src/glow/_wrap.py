__all__ = ['wrap']

from collections.abc import Callable, Generator, Iterator
from itertools import count
from typing import Any, Protocol, Self

from wrapt import ObjectProxy


def wrap[**P, R](func: Callable[P, R], wrapper: '_Wrapper') -> Callable[P, R]:
    return _Callable(func, wrapper)


class _Wrapper(Protocol):
    @property
    def calls(self) -> count: ...

    # This one start right after function was called,
    # and stops right before it's called again.
    # Usage:
    #   fn(*args, **kwargs)
    #   resume = wrapper.suspend()
    #   ...
    #   resume()
    #   fn(*args, **kwargs)
    def suspend(self) -> Callable[[], None]: ...

    # This one start right before function was called,
    # and stops right after it returned.
    # Usage:
    #   return wrapper(fn, *args, **kwargs)
    def __call__[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwds: P.kwargs
    ) -> R: ...


class _Proxy[T](ObjectProxy):  # type: ignore[misc]
    __wrapped__: T
    _self_wrapper: _Wrapper

    def __init__(self, wrapped: T, wrapper: _Wrapper) -> None:
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
        next(self._self_wrapper.calls)
        r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        return _wrap(r, self._self_wrapper)


class _BoundCallable[**P, R](_Callable[P, R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


# ----------------------------------- sync -----------------------------------


class _WrapStop:
    _self_wrapper: _Wrapper

    def _wrap_stop[**P, R](
        self, op: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        try:
            ret = self._self_wrapper(op, *args, **kwargs)
        except StopIteration as stop:
            stop.value = _wrap(stop.value, self._self_wrapper)
            raise
        else:
            return _wrap(ret, self._self_wrapper)


class _IterNext[Y](_WrapStop):
    __wrapped__: Iterator[Y]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.__next__)


class _SendThrowClose[Y, S](_WrapStop):
    __wrapped__: Generator[Y, S, Any]

    def send(self, value: S, /) -> Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.send, value)

    def throw(self, value: BaseException, /) -> Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.throw, value)

    def close(self) -> Any | None:  # ! + time
        return self._wrap_stop(self.__wrapped__.close)


class _Iterator[Y](_IterNext[Y], _Proxy[Iterator[Y]]):
    pass


class _Generator[Y, S, R](
    _SendThrowClose[Y, S], _IterNext[Y], _Proxy[Generator[Y, S, R]]
):
    pass


def _wrap[T](r: T, wrapper: _Wrapper) -> T:
    # function & generator
    # are distinguishable only by their result
    match r:
        # `r` comes from generator function (types.GeneratorType)
        # or compatible
        case Generator():
            # return _gen(r, wrapper)
            return _Generator(r, wrapper)

        # `r` comes from function that returned iterator
        case Iterator():
            return _Iterator(r, wrapper)

    return r
