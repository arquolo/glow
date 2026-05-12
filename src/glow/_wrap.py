__all__ = ['wrap']

import types
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from itertools import count
from typing import Any, Protocol, Self

from wrapt import ObjectProxy

from ._types import Coro, Get


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
    def suspend(self) -> Get[None]: ...

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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # !! time
        # patch & record fn.__call__
        next(self._self_wrapper.calls)
        r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        return _wrap(r, self._self_wrapper)


class _BoundCallable[**P, R](_Callable[P, R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


# ----------------------------------- sync -----------------------------------


class _Iterator[Y](_Proxy[Iterator[Y]]):
    # __iter__: () -> Iterator[Y]
    # __next__: () -> Y

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:  # ! + time
        ret = self._self_wrapper(self.__wrapped__.__next__)
        return _wrap(ret, self._self_wrapper)


class _Generator[Y, S, R](_Proxy[Generator[Y, S, R]]):
    # __iter__: () -> Generator[Y, S, R]
    # __next__: () -> Y
    # send: (S) -> Y
    # throw: (BaseException) -> Y
    # close: () -> None

    def __init__(self, wrapped: Generator[Y, S, R], wrapper: _Wrapper) -> None:
        super().__init__(wrapped, wrapper)
        self._self_resume: Get[None] | None = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:  # !! time
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.__next__)
        # ! maybe do this in Coroutine instead ?
        if isinstance(ret, Awaitable):
            self._self_resume = self._self_wrapper.suspend()
        return _wrap(ret, self._self_wrapper)

    def send(self, value: S, /) -> Y:  # ? time>>0
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.send, value)
        if isinstance(ret, Awaitable):
            self._self_resume = self._self_wrapper.suspend()
        # return _wrap(ret, self._self_wrapper)  # ! not used
        return ret

    def throw(self, value: BaseException, /) -> Y:  # ? time>>0
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.throw, value)
        if isinstance(ret, Awaitable):
            self._self_resume = self._self_wrapper.suspend()
        # return _wrap(ret, self._self_wrapper)  # ! not used
        return ret

    def close(self) -> R | None:  # !! time
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        return self._self_wrapper(self.__wrapped__.close)


def _as_gen[Y, S, R](
    gen: Generator[Y, S, R], wrapper: _Wrapper
) -> Generator[Y, S, R]:
    assert iter(gen) is gen
    try:
        item = wrapper(gen.__next__)
        while True:
            try:
                send = yield item
            except GeneratorExit:
                if (ret := wrapper(gen.close)) is not None:
                    return ret
                raise
            except BaseException as exc:  # noqa: BLE001
                item = wrapper(gen.throw, exc)
            else:
                item = (
                    wrapper(gen.__next__)
                    if send is None
                    else wrapper(gen.send, send)
                )
    except StopIteration as e:
        return e.value


# ---------------------------------- async -----------------------------------


@types.coroutine
def _await[Y](x: Y) -> Generator[Y, Any, Any]:
    yield x


class _Awaitable[R](_Proxy[Awaitable[R]]):
    # I.e. asyncio.Future, asyncio.Task, e.t.c.

    def __await__(self) -> Generator[Any, Any, R]:
        it = self._self_wrapper(self.__wrapped__.__await__)
        return _wrap(it, self._self_wrapper)


class _Coroutine[Y, S, R](_Proxy[Coroutine[Y, S, R]]):
    # __await__: () -> Generator[Any, Any, R]
    # send: (S) -> Y
    # throw: (BaseException) -> Y
    # close: () -> None

    def __await__(self) -> Generator[None, None, R]:  # ? time>>0
        # ! do-not-track - it's instant
        # TODO: which also are?
        it = self._self_wrapper(self.__wrapped__.__await__)
        return _wrap(it, self._self_wrapper)  # generator

    def send(self, value: S, /) -> Y:  # !! time
        ret = self._self_wrapper(self.__wrapped__.send, value)
        return _wrap(ret, self._self_wrapper)

    def throw(self, value: BaseException, /) -> Y:  # !! time
        ret = self._self_wrapper(self.__wrapped__.throw, value)
        return _wrap(ret, self._self_wrapper)

    def close(self) -> Any | None:  # ? time>>0
        return self._self_wrapper(self.__wrapped__.close)


async def _as_coroutine[R](coro: Coro[R], wrapper: _Wrapper) -> R:
    try:
        fut = wrapper(coro.send, None)
        while True:
            resume = wrapper.suspend()
            try:
                send = await _await(fut)  # ok - never throws StopIteration
            except GeneratorExit:
                resume()
                wrapper(coro.close)
                raise
            except BaseException as exc:  # noqa: BLE001
                resume()
                fut = wrapper(coro.throw, exc)
            else:
                resume()
                fut = wrapper(coro.send, send)
    except StopIteration as e:
        return e.value


class _CoroutineGenerator[Y, S, R](
    _Proxy[Coroutine[Y, S, R] | Generator[Y, S, R]]
):
    # __await__: () -> Self
    # __iter__: () -> Self
    # __next__: () -> Y
    # send: (S) -> Y
    # throw: (BaseException) -> Y
    # close: () -> R

    def __init__(
        self,
        wrapped: Coroutine[Y, S, R] | Generator[Y, S, R],
        wrapper: _Wrapper,
    ) -> None:
        super().__init__(wrapped, wrapper)
        self._self_resume: Get[None] | None = None

    def __await__(self) -> Self:
        return self

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:  # !! time
        assert isinstance(self.__wrapped__, Iterator)
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.__next__)
        self._self_resume = self._self_wrapper.suspend()
        return _wrap(ret, self._self_wrapper)

    def send(self, value: S, /) -> Y:  # ? time>>0
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.send, value)
        self._self_resume = self._self_wrapper.suspend()
        return _wrap(ret, self._self_wrapper)

    def throw(self, value: BaseException, /) -> Y:  # ? time>>0
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        ret = self._self_wrapper(self.__wrapped__.throw, value)
        self._self_resume = self._self_wrapper.suspend()
        return _wrap(ret, self._self_wrapper)

    def close(self) -> R | None:  # ? time>>0
        if self._self_resume:
            self._self_resume()
            self._self_resume = None
        return self._self_wrapper(self.__wrapped__.close)


class _AsyncIterator[Y](_Proxy[AsyncIterator[Y]]):
    # __aiter__: () -> Self
    # __anext__: () -> Awaitable[Y]

    def __aiter__(self) -> Self:
        return self

    def __anext__(self) -> Awaitable[Y]:
        return _wrap(self.__wrapped__.__anext__(), self._self_wrapper)


async def _as_asyncgen[Y, S](
    asyncgen: AsyncGenerator[Y, S], wrapper: _Wrapper
) -> AsyncGenerator[Y, S]:
    assert asyncgen.__aiter__() is asyncgen
    coro = asyncgen.__anext__()

    while True:
        try:
            item = await _wrap(coro, wrapper)  # coroutine
        except StopAsyncIteration:
            return

        try:
            send = yield item
        except GeneratorExit:
            await _wrap(asyncgen.aclose(), wrapper)  # coroutine
            raise

        except BaseException as exc:  # noqa: BLE001
            coro = asyncgen.athrow(exc)
        else:
            if send is None:
                coro = asyncgen.__anext__()
            else:
                coro = asyncgen.asend(send)


class _AsyncGenerator[Y, S](_Proxy[AsyncGenerator[Y, S]]):
    # __aiter__: () -> Self
    # __anext__: () -> Coro[Y]
    # asend: (S) -> Coro[Y]
    # athrow: (BaseException) -> Coro[Y]
    # aclose: () -> Coro[None]

    def __aiter__(self) -> Self:
        return self

    def __anext__(self) -> Coro[Y]:
        return _wrap(self.__wrapped__.__anext__(), self._self_wrapper)

    def asend(self, value: S, /) -> Coro[Y]:
        return _wrap(self.__wrapped__.asend(value), self._self_wrapper)

    def athrow(self, value: BaseException, /) -> Coro[Y]:
        return _wrap(self.__wrapped__.athrow(value), self._self_wrapper)

    def aclose(self) -> Coro[None]:
        return _wrap(self.__wrapped__.aclose(), self._self_wrapper)


# -------------------------------- decoration --------------------------------


def _wrap[T](obj: T, wrapper: _Wrapper) -> T:
    # from glow import whereami
    # print(whereami(0, -30), type(obj), obj)
    # function, generator, coroutine & async generator
    # are distinguishable only by their result
    if isinstance(obj, _Proxy):
        return obj
    for tps, deco in _wrappers.items():
        if all(isinstance(obj, tp) for tp in tps):
            return deco(obj, wrapper)
    return obj


# NOTE: order is crucial
type Apply[T] = Callable[[T, _Wrapper], T]

_wrappers: dict[tuple[type, ...], Apply] = {
    (Coroutine, Generator): _CoroutineGenerator,
    #
    # types.AsyncGeneratorType (or compatible) from async generator func
    #   async for item in obj:
    #       ...
    # is syntaxic sugar around:
    #   while True:
    #       item = obj.__anext__().__await__().__next__()
    # with true computation in `__next__()`
    # (types.AsyncGeneratorType,): _as_asyncgen,
    (AsyncGenerator,): _AsyncGenerator,
    #
    # async iterator from function
    #   async for item in obj:
    #       ...
    # is syntaxic sugar around:
    #   while True:
    #       item = obj.__anext__().__await__().__next__()
    # with true computation in `__next__()`
    (AsyncIterator,): _AsyncIterator,
    #
    # types.CoroutineType (or compatible) from coroutine function
    # `await obj` is same as `obj.__await__().__next__()`
    # with true computation in `__next__()`
    # (types.CoroutineType,): _as_coroutine,
    (Coroutine,): _Coroutine,
    #
    # Future-like from function
    (Awaitable,): _Awaitable,
    #
    # types.GeneratorType (or compatible) comes from generator function
    # we need to wrap `send()`, `throw()` & `close()`
    # (types.GeneratorType,): _as_gen,
    (Generator,): _Generator,
    #
    # iterator from function
    #   for item in obj:
    #       ...
    # is syntaxic sugar around:
    #   while True:
    #       item = obj.__next__()
    # with true computation in `__next__()`
    (Iterator,): _Iterator,
}
