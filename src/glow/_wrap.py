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
from typing import Any, Never, Protocol, Self, cast

try:
    from wrapt import BaseObjectProxy as ObjectProxy  # wrapt>=2.0
except ImportError:
    from wrapt import ObjectProxy

from ._dev import hide_frame
from ._types import Coro, Get


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


class _SuspendProxy[T](ObjectProxy):
    __wrapped__: T

    def __init__(self, wrapped: T, wrapper: Wrapper) -> None:
        super().__init__(wrapped)
        self._self_wrapper = wrapper
        self._self_resume: Get[None] | None = None

    def _resume(self) -> None:
        if self._self_resume:
            self._self_resume()
            self._self_resume = None

    def _suspend(self) -> None:
        self._self_resume = self._self_wrapper.suspend()


class _Proxy[T](_SuspendProxy[T]):
    def __init__(
        self, wrapped: T, wrapper: Wrapper, suspend: bool = False
    ) -> None:
        super().__init__(wrapped, wrapper)
        self._self_suspend = suspend

    def _suspend(self) -> None:
        if self._self_suspend:
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
            case _Proxy() | _SuspendProxy():
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


class _IterNext[Y]:
    __wrapped__: Iterator[Y]
    _self_wrapper: Wrapper
    _resume: Callable[..., None]
    _suspend: Callable[..., None]

    def __iter__(self) -> Iterator[Y]:
        itr = self._self_wrapper(self.__wrapped__.__iter__)
        return self if itr is self.__wrapped__ else itr

    def __next__(self) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.__next__)
        self._suspend()
        return ret


class _SendThrowClose[Y, S, R]:
    __wrapped__: Generator[Y, S, R] | Coroutine[Y, S, R]
    _self_wrapper: Wrapper
    _resume: Callable[..., None]
    _suspend: Callable[..., None]

    def send(self, value: S, /) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.send, value)
        self._suspend()
        return ret

    def throw(self, *args) -> Y:
        self._resume()
        with hide_frame:
            ret = self._self_wrapper(self.__wrapped__.throw, *args)
        self._suspend()
        return ret

    def close(self) -> R | None:
        self._resume()
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.close)


class _Await[R]:
    __wrapped__: Awaitable[R]
    _self_wrapper: Wrapper
    _resume: Callable[..., None]
    _suspend: Callable[..., None]

    def __await__(self) -> Generator[Any, Any, R]:
        with hide_frame:
            it = self._self_wrapper(self.__wrapped__.__await__)
        if it is self.__wrapped__ and isinstance(self, _IterNext):  # type: ignore[comparison-overlap]
            return self
        return _wrap_iter(it, self._self_wrapper, suspend=True)  # type: ignore[return-value]


class _Iterator[Y](_IterNext[Y], _Proxy[Iterator[Y]]):
    pass


class _IteratorSuspend[Y](_IterNext[Y], _SuspendProxy[Iterator[Y]]):
    def send(self, value: Any, /) -> Y:
        self._resume()
        with hide_frame:
            if value is None:
                ret = self._self_wrapper(self.__wrapped__.__next__)
            else:
                ret = self._self_wrapper(self.__wrapped__.send, value)  # type: ignore[attr-defined]
        self._suspend()
        return ret

    def throw(self, *args) -> Y:
        self._resume()
        if (throw := getattr(self.__wrapped__, 'throw', None)) is not None:
            with hide_frame:
                ret = self._self_wrapper(throw, *args)
            self._suspend()
            return ret

        with hide_frame:
            _yield_never().throw(*args)
        raise AssertionError

    def close(self) -> Any | None:
        self._resume()
        if (close := getattr(self.__wrapped__, 'close', None)) is not None:
            with hide_frame:
                return self._self_wrapper(close)
        return None


class _Generator[Y, S, R](
    _SendThrowClose[Y, S, R], _IterNext[Y], _Proxy[Generator[Y, S, R]]
):
    __wrapped__: Generator[Y, S, R]


class _FutureLike[R](_Await[R], _SuspendProxy[Awaitable[R]]):
    pass


class _Coroutine[Y, S, R](
    _SendThrowClose[Y, S, R], _Await[R], _SuspendProxy[Coroutine[Y, S, R]]
):
    __wrapped__: Coroutine[Y, S, R]


class _CoroutineGenerator[Y, S, R](
    _IterNext[Y],
    _SendThrowClose[Y, S, R],
    _Await[R],
    _SuspendProxy[Coroutine[Y, S, R] | Generator[Y, S, R]],
):
    __wrapped__: Coroutine[Y, S, R] | Generator[Y, S, R]  # type: ignore[assignment]


class _AsyncIterator[Y](_Proxy[AsyncIterator[Y]]):
    def __aiter__(self) -> AsyncIterator[Y]:
        with hide_frame:
            aitr = self._self_wrapper(self.__wrapped__.__aiter__)
        if aitr is self.__wrapped__:
            return self
        if isinstance(aitr, _Proxy | _SuspendProxy):
            return aitr
        if isinstance(aitr, AsyncIterator):
            return _wrap_asynciter(aitr, self._self_wrapper)
        return aitr

    def __anext__(self) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.__anext__)
            return _wrap_awaitable(aw, self._self_wrapper)


class _AsyncGenerator[Y, S](_AsyncIterator[Y]):
    __wrapped__: AsyncGenerator[Y, S]

    def asend(self, value: S, /) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.asend, value)
            return _wrap_awaitable(aw, self._self_wrapper)

    def athrow(self, *args) -> Awaitable[Y]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.athrow, *args)
            return _wrap_awaitable(aw, self._self_wrapper)

    def aclose(self) -> Awaitable[None]:
        with hide_frame:
            aw = self._self_wrapper(self.__wrapped__.aclose)
            return _wrap_awaitable(aw, self._self_wrapper)


# ------------------------------ *type wrappers ------------------------------


def _gen[Y, S, R](
    gen: types.GeneratorType[Y, S, R],
    wrapper: Wrapper,
    suspend: bool = False,
) -> Generator[Y, S, R]:
    assert iter(gen) is gen
    op: Get[Y] = gen.__next__
    try:
        while True:
            with hide_frame:
                item = wrapper(op)

            resume = wrapper.suspend() if suspend else None
            try:
                try:
                    with hide_frame:
                        send = yield item
                finally:
                    if resume is not None:
                        resume()

            except GeneratorExit as exc:
                if suspend:
                    with hide_frame:
                        wrapper(gen.close)
                    raise
                op = partial(gen.throw, exc)
            except BaseException as exc:  # noqa: BLE001
                op = partial(gen.throw, exc)
            else:
                op = gen.__next__ if send is None else partial(gen.send, send)

    except StopIteration as e:
        return e.value


@types.coroutine
def _await[Y, S](y: Y, sent: list[S]) -> Generator[Y, S, Any]:
    sent.append((yield y))


async def _coroutine[R](
    coro: types.CoroutineType[Any, Any, R], wrapper: Wrapper
) -> R:
    genex: GeneratorExit | None = None
    op: Get = partial(coro.send, None)
    try:
        while True:
            with hide_frame:
                ret = wrapper(op)  # throws anything

                if genex:
                    # raise RuntimeError('coroutine ignored GeneratorExit')
                    raise genex

            if getattr(ret, '_asyncio_future_blocking', None):  # Future
                sent = [None]
                ret._asyncio_future_blocking = False
            else:
                sent = []
                ret = _await(ret, sent)

            try:
                resume = wrapper.suspend()
                try:
                    with hide_frame:
                        await ret  # never throws StopIteration
                finally:
                    resume()
            except GeneratorExit as exc:
                genex = exc
                op = coro.close
            except BaseException as exc:  # noqa: BLE001
                op = partial(coro.throw, exc)
            else:
                op = partial(coro.send, sent[0])

    except StopIteration as e:
        return e.value


async def _asyncgen[Y, S](
    asyncgen: types.AsyncGeneratorType[Y, S], wrapper: Wrapper
) -> AsyncGenerator[Y, S]:
    assert aiter(asyncgen) is asyncgen
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


def _wrap_iter[Y](
    it: Iterator[Y], wrapper: Wrapper, suspend: bool = False
) -> Iterator[Y]:
    if isinstance(it, Generator):  # + send, throw, close
        if isinstance(it, types.GeneratorType):  # genfuncs
            return _gen(it, wrapper, suspend)
        if isinstance(it, Coroutine):  # + __await__
            return _CoroutineGenerator(it, wrapper)
        return _Generator(it, wrapper, suspend)  # user's generator
    if suspend:
        return _IteratorSuspend(it, wrapper)  # user's iterator
    return _Iterator(it, wrapper)  # user's iterator


def _wrap_awaitable[R](aw: Awaitable[R], wrapper: Wrapper) -> Awaitable[R]:
    if isinstance(aw, Coroutine):  # + send, throw, close
        if isinstance(aw, types.CoroutineType):  # corofuncs
            cr = _coroutine(aw, wrapper)
            weakref.finalize(cr, aw.close)  # ... was never awaited
            return cr
        if isinstance(aw, Generator):  # + __iter__, __next__
            return _CoroutineGenerator(aw, wrapper)
        return _Coroutine(aw, wrapper)  # user's coroutine
    return _FutureLike(aw, wrapper)


def _wrap_asynciter[R](
    aitr: AsyncIterator[R], wrapper: Wrapper
) -> AsyncIterator[R]:
    if isinstance(aitr, AsyncGenerator):  # + asend, athrow, aclose
        if isinstance(aitr, types.AsyncGeneratorType):  # asyncgen funcs
            return _asyncgen(aitr, wrapper)
        return _AsyncGenerator(aitr, wrapper)  # user's asyncgen
    return _AsyncIterator(aitr, wrapper)  # user's asynciter


def _yield_never() -> Generator[Never]:
    return
    yield
