__all__ = ['wrap']

import types
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from functools import partial
from typing import Any, Protocol, Self, cast

try:
    from wrapt import BaseObjectProxy as ObjectProxy  # wrapt>=2.0
except ImportError:
    from wrapt import ObjectProxy

from ._dev import hide_frame
from ._types import Coro, Get, HasClose, HasSend, HasThrow


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
    # NOTE: suspend should be recorded only on `resume()` call.
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


class _AwProxy[T](_Proxy[T]):
    def __init__(self, wrapped: T, wrapper: Wrapper) -> None:
        super().__init__(wrapped, wrapper)
        self._self_resume: Get[None] | None = None

    def _resume(self) -> None:
        if self._self_resume:
            self._self_resume()
            self._self_resume = None

    def _suspend(self) -> None:
        self._self_resume = self._self_wrapper.suspend()


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

        # function, generator, coroutine & async generator
        # are distinguishable only by their result
        match r:
            case _Proxy():
                return r
            case AsyncIterator():  # __aiter__/__anext__
                return cast('R', _wrap_asynciter(r, self._self_wrapper))
            case Awaitable():  # __await__
                return cast('R', _wrap_awaitable(r, self._self_wrapper))
            case Iterator():  # __iter__/__next__
                return cast('R', _wrap_iter(r, self._self_wrapper))
        return r


class _BoundCallable[**P, R](_Callable[P, R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


# ---------------------------------- bases -----------------------------------


class _Iterator[Y](_Proxy[Iterator[Y]]):
    def __iter__(self) -> Iterator[Y]:
        itr = self._self_wrapper(self.__wrapped__.__iter__)
        return self if itr is self.__wrapped__ else itr

    def __next__(self) -> Y:
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.__next__)


class _Generator[Y, S, R](_Iterator[Y]):
    def send(self: _Proxy[Generator[Y, S, R]], value: S, /) -> Y:
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.send, value)

    def throw(self: _Proxy[Generator[Y, S, R]], *args) -> Y:
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.throw, *args)

    def close(self: _Proxy[Generator[Y, S, R]]) -> R | None:
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.close)


class _AwIterator[Y](_Iterator[Y], _AwProxy[Iterator[Y]]):
    def __next__(self) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.__next__)
        self._suspend()
        return ret


class _AwSend[Y, S](_AwProxy):
    def send(self: _AwProxy[HasSend[Y, S]], value: S, /) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.send, value)
        self._suspend()
        return ret


class _AwThrow[Y](_AwProxy):
    def throw(self: _AwProxy[HasThrow[Y]], *args) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.throw, *args)
        self._suspend()
        return ret


class _AwClose[R](_AwProxy):
    def close(self: _AwProxy[HasClose[R]]) -> R | None:
        self._resume()
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.close)


_table: dict[tuple[bool, bool, bool], type[_AwProxy]] = {
    (bool(s), bool(t), bool(c)): type(
        '_AwIterator' + ''.join(tp.__name__[3:] for tp in s + t + c),
        (*s, *t, *c, _AwIterator),
        {},
    )
    for s in ([], [_AwSend])
    for t in ([], [_AwThrow])
    for c in ([], [_AwClose])
}
_AwGenerator = _table[True, True, True]


class _Awaitable[R](_AwProxy[Awaitable[R]]):
    def __await__(self) -> Generator[Any, Any, R]:
        with hide_frame:
            it = self._self_wrapper(self.__wrapped__.__await__)
        if it is self.__wrapped__ and isinstance(self, _AwIterator):  # type: ignore[comparison-overlap]
            return self  # type: ignore[return-value]
        return _wrap_aw_iter(it, self._self_wrapper)  # type: ignore[return-value]


class _Coroutine[Y, S, R](
    _AwClose[R], _AwThrow[Y], _AwSend[Y, S], _Awaitable[R]
):
    pass


class _CoroutineGenerator[Y, S, R](_Coroutine[Y, S, R], _AwIterator[Y]):
    pass


class _AsyncIterator[Y](_Proxy[AsyncIterator[Y]]):
    def __aiter__(self) -> AsyncIterator[Y]:
        with hide_frame:
            aitr = self._self_wrapper(self.__wrapped__.__aiter__)
        if aitr is self.__wrapped__:
            return self
        if isinstance(aitr, _Proxy):
            return aitr
        if isinstance(aitr, AsyncIterator):
            return _wrap_asynciter(aitr, self._self_wrapper)
        return aitr

    def __anext__(self) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.__anext__)
            return _wrap_awaitable(aw, self._self_wrapper)


class _AsyncGenerator[Y, S](_AsyncIterator[Y]):
    def asend(self: _Proxy[AsyncGenerator[Y, S]], value: S, /) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.asend, value)
            return _wrap_awaitable(aw, self._self_wrapper)

    def athrow(self: _Proxy[AsyncGenerator[Y, S]], *args) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.athrow, *args)
            return _wrap_awaitable(aw, self._self_wrapper)

    def aclose(self: _Proxy[AsyncGenerator[Y, S]]) -> Awaitable[None]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.aclose)
            return _wrap_awaitable(aw, self._self_wrapper)


# ------------------------------ *type wrappers ------------------------------


def _gen[Y, S, R](
    gen: types.GeneratorType[Y, S, R], wrapper: Wrapper
) -> Generator[Y, S, R]:
    op: Get[Y] = gen.__next__
    try:
        while True:
            with hide_frame:
                item = wrapper(op)

            try:
                with hide_frame:
                    send = yield item
            except BaseException as exc:  # noqa: BLE001
                op = partial(gen.throw, exc)
            else:
                op = gen.__next__ if send is None else partial(gen.send, send)

    except StopIteration as e:
        return e.value


def _gen_aw[Y, S, R](
    gen: types.GeneratorType[Y, S, R], wrapper: Wrapper
) -> Generator[Y, S, R]:
    op: Get[Y] = gen.__next__
    try:
        while True:
            with hide_frame:
                item = wrapper(op)

            resume = wrapper.suspend()
            try:
                try:
                    with hide_frame:
                        send = yield item
                finally:
                    resume()
            except GeneratorExit:
                with hide_frame:
                    wrapper(gen.close)
                raise
            except BaseException as exc:  # noqa: BLE001
                op = partial(gen.throw, exc)
            else:
                op = gen.__next__ if send is None else partial(gen.send, send)

    except StopIteration as e:
        return e.value


@types.coroutine
def _await[S](value: None, sent: list[S]) -> Generator[None, S]:
    # For asyncio's event loop `y` should be None,
    # otherwise `await _await(...)` will trigger: `Task got bad yield: ...`.
    # Other implementations of event loop could support more types.
    sent.append((yield value))


async def _coroutine[R](
    coro: types.CoroutineType[Any, Any, R], wrapper: Wrapper
) -> R:
    genex: GeneratorExit | None = None
    op: Get = partial(coro.send, None)
    try:
        while True:
            with hide_frame:
                yielded = wrapper(op)  # throws anything

                if genex:
                    raise genex

            sent: list[None] = []
            try:
                resume = wrapper.suspend()
                try:
                    # Future
                    if getattr(yielded, '_asyncio_future_blocking', None):
                        yielded._asyncio_future_blocking = False
                        sent.append(None)
                        with hide_frame:
                            await yielded  # Won't stopiter
                    # `None` for asyncio
                    else:
                        with hide_frame:
                            await _await(yielded, sent)  # Won't stopiter
                finally:
                    resume()
            except GeneratorExit as exc:
                genex = exc
                op = coro.close
            except BaseException as exc:  # noqa: BLE001
                op = partial(coro.throw, exc)
            else:
                assert sent
                op = partial(coro.send, *sent)

    except StopIteration as e:
        return e.value


async def _asyncgen[Y, S](
    asyncgen: types.AsyncGeneratorType[Y, S], wrapper: Wrapper
) -> AsyncGenerator[Y, S]:
    op: Get[Coro[Y]] = asyncgen.__anext__

    while True:
        try:
            with hide_frame:  # coroutine
                item = await _wrap_awaitable(wrapper(op), wrapper)
        except StopAsyncIteration:
            return

        try:
            with hide_frame:
                send = yield item
        except BaseException as exc:  # noqa: BLE001
            op = partial(asyncgen.athrow, exc)
        else:
            op = (
                asyncgen.__anext__
                if send is None
                else partial(asyncgen.asend, send)
            )


# -------------------------------- decoration --------------------------------


def _wrap_iter[Y](it: Iterator[Y], wrapper: Wrapper) -> Iterator[Y]:
    if isinstance(it, Generator):  # + send, throw, close
        if isinstance(it, types.GeneratorType):  # genfuncs
            return _gen(it, wrapper)
        if isinstance(it, Coroutine):  # + __await__
            return _CoroutineGenerator(it, wrapper)
        return _Generator(it, wrapper)  # user's generator
    return _Iterator(it, wrapper)  # user's iterator


def _wrap_aw_iter[Y](it: Iterator[Y], wrapper: Wrapper) -> Iterator[Y]:
    if isinstance(it, Generator):  # + send, throw, close
        if isinstance(it, types.GeneratorType):  # genfuncs
            return _gen_aw(it, wrapper)
        if isinstance(it, Coroutine):  # + __await__
            return _CoroutineGenerator(it, wrapper)
        return _AwGenerator(it, wrapper)  # user's generator

    # user's iterator
    return _table[
        getattr(it, 'send', None) is not None,
        getattr(it, 'throw', None) is not None,
        getattr(it, 'close', None) is not None,
    ](it, wrapper)


def _wrap_awaitable[R](aw: Awaitable[R], wrapper: Wrapper) -> Awaitable[R]:
    if isinstance(aw, Coroutine):  # + send, throw, close
        if isinstance(aw, types.CoroutineType):  # corofuncs
            cr = _coroutine(aw, wrapper)
            weakref.finalize(cr, aw.close)  # ... was never awaited
            return cr
        if isinstance(aw, Generator):  # + __iter__, __next__
            return _CoroutineGenerator(aw, wrapper)
        return _Coroutine(aw, wrapper)  # user's coroutine
    return _Awaitable(aw, wrapper)


def _wrap_asynciter[R](
    aitr: AsyncIterator[R], wrapper: Wrapper
) -> AsyncIterator[R]:
    if isinstance(aitr, AsyncGenerator):  # + asend, athrow, aclose
        if isinstance(aitr, types.AsyncGeneratorType):  # asyncgen funcs
            return _asyncgen(aitr, wrapper)
        return _AsyncGenerator(aitr, wrapper)  # user's asyncgen
    return _AsyncIterator(aitr, wrapper)  # user's asynciter
