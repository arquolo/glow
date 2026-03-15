__all__ = ['wrap']

from collections.abc import Callable, Generator, Iterator
from itertools import count
from typing import Any, Generic, ParamSpec, Protocol, Self, TypeVar

from wrapt import ObjectProxy

from ._types import Get

_T = TypeVar('_T')
_Y = TypeVar('_Y')
_S = TypeVar('_S')
_R = TypeVar('_R')
_P = ParamSpec('_P')


def wrap(func: Callable[_P, _R], wrapper: '_Wrapper') -> Callable[_P, _R]:
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
    def suspend(self) -> Get[None]: ...

    # This one start right before function was called,
    # and stops right after it returned.
    # Usage:
    #   return wrapper(fn, *args, **kwargs)
    def __call__(
        self, fn: Callable[_P, _R], /, *args: _P.args, **kwds: _P.kwargs
    ) -> _R: ...


class _Proxy(ObjectProxy, Generic[_T]):  # type: ignore[misc]
    __wrapped__: _T
    _self_wrapper: _Wrapper

    def __init__(self, wrapped: _T, wrapper: _Wrapper) -> None:
        super().__init__(wrapped)
        self._self_wrapper = wrapper


class _Callable(_Proxy[Callable[_P, _R]]):
    def __get__(
        self, instance: object, owner: type | None
    ) -> '_BoundCallable':
        fn = self.__wrapped__.__get__(instance, owner)
        return _BoundCallable(fn, self._self_wrapper)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        # patch & record fn.__call__
        next(self._self_wrapper.calls)
        r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        return _wrap(r, self._self_wrapper)


class _BoundCallable(_Callable[_P, _R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


# ----------------------------------- sync -----------------------------------


class _WrapStop:
    _self_wrapper: _Wrapper

    def _wrap_stop(
        self, op: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        try:
            ret = self._self_wrapper(op, *args, **kwargs)
        except StopIteration as stop:
            stop.value = _wrap(stop.value, self._self_wrapper)
            raise
        else:
            return _wrap(ret, self._self_wrapper)


class _IterNext(_WrapStop, Generic[_Y]):
    __wrapped__: Iterator[_Y]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.__next__)


class _SendThrowClose(_WrapStop, Generic[_Y, _S]):
    __wrapped__: Generator[_Y, _S, Any]

    def send(self, value: _S, /) -> _Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.send, value)

    def throw(self, value: BaseException, /) -> _Y:  # ! + time
        return self._wrap_stop(self.__wrapped__.throw, value)

    def close(self) -> Any | None:  # ! + time
        return self._wrap_stop(self.__wrapped__.close)


class _Iterator(_IterNext[_Y], _Proxy[Iterator[_Y]]):
    pass


class _Generator(
    _SendThrowClose[_Y, _S], _IterNext[_Y], _Proxy[Generator[_Y, _S, _R]]
):
    pass


def _wrap(r: _T, wrapper: _Wrapper) -> _T:
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
