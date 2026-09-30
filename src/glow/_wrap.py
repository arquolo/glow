__all__ = ['wrap']

import asyncio
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
from typing import Any, Protocol, Self

try:
    from wrapt import BaseObjectProxy as ObjectProxy  # wrapt>=2.0
except ImportError:
    from wrapt import ObjectProxy

from ._dev import hide_frame
from ._types import Coro, Get

_OP_FORK_STOPITER = True
_OP_FUNC = True  # py_anext.<locals>.anext_impl was never awaited


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
        self._self_resume: Get[None] | None = None

    def _resume(self) -> None:
        if not self._self_resume:
            return
        self._self_resume()
        self._self_resume = None

    def _suspend(self, obj) -> None:
        if isinstance(obj, _Proxy) or not isinstance(obj, asyncio.Future):
            return
        if self._self_resume:  # reentrancy
            return
        # TODO: fix false positive for generator yielding futures
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
        self._resume()
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.__next__)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        self._suspend(ret)
        return _wrap(ret, self._self_wrapper)


class _CoroLike[Y, S, R](_Proxy[Generator[Y, S, R] | Coroutine[Y, S, R]]):
    def send(self, value: S, /) -> Y:
        self._resume()
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.send, value)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        self._suspend(ret)
        return _wrap(ret, self._self_wrapper)

    def throw(self, *args) -> Y:
        self._resume()
        with hide_frame:
            try:
                ret = self._self_wrapper(self.__wrapped__.throw, *args)
            except StopIteration as stop:
                _wrap(stop, self._self_wrapper)
                raise
        self._suspend(ret)
        return _wrap(ret, self._self_wrapper)

    def close(self) -> R | None:
        self._resume()
        with hide_frame:
            return self._self_wrapper(self.__wrapped__.close)


class _Generator[Y, S, R](_CoroLike[Y, S, R], _Iterator[Y]):
    pass


class _FutureLike[R](_Proxy[Awaitable[R]]):
    def __await__(self) -> Generator[Any, Any, R]:
        with hide_frame:
            gen = self._self_wrapper(self.__wrapped__.__await__)
        if gen is self.__wrapped__:  # type: ignore[comparison-overlap]
            assert isinstance(self, Generator)
            return self
        return _wrap(gen, self._self_wrapper)


class _Coroutine[Y, S, R](_CoroLike[Y, S, R], _FutureLike[R]):
    pass


class _CoroutineGenerator[Y, S, R](_Generator[Y, S, R], _FutureLike[R]):
    pass


class _AsyncIterator[Y](_Proxy[AsyncIterator[Y]]):
    def __aiter__(self) -> AsyncIterator[Y]:
        with hide_frame:
            aitr = self._self_wrapper(self.__wrapped__.__aiter__)
        if aitr is self.__wrapped__:
            return self
        return _wrap(aitr, self._self_wrapper)

    def __anext__(self) -> Awaitable[Y]:
        with hide_frame:
            return _wrap(self.__wrapped__.__anext__(), self._self_wrapper)


class _AsyncGenerator[Y, S](_AsyncIterator[Y]):
    __wrapped__: AsyncGenerator[Y, S]

    def asend(self, value: S, /) -> Coro[Y]:
        with hide_frame:
            return _wrap(self.__wrapped__.asend(value), self._self_wrapper)

    def athrow(self, *args) -> Coro[Y]:
        with hide_frame:
            return _wrap(self.__wrapped__.athrow(*args), self._self_wrapper)

    def aclose(self) -> Coro[None]:
        with hide_frame:
            return _wrap(self.__wrapped__.aclose(), self._self_wrapper)


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
        return _wrap(e, wrapper).value


async def _asyncgen[Y, S](
    asyncgen: types.AsyncGeneratorType[Y, S], wrapper: Wrapper
) -> AsyncGenerator[Y, S]:
    assert aiter(asyncgen) is asyncgen
    op: Get[Coro[Y]] = asyncgen.__anext__

    while True:
        try:
            with hide_frame:
                item = await _wrap(op(), wrapper)  # coroutine
        except StopAsyncIteration:
            return

        try:
            with hide_frame:
                send = yield _wrap(item, wrapper)
        except BaseException as exc:  # noqa: BLE001
            op = partial(asyncgen.athrow, exc)
        else:
            op = (
                asyncgen.__anext__
                if send is None
                else partial(asyncgen.asend, send)
            )


# -------------------------------- decoration --------------------------------


def _wrap[T](r: T, wrapper: Wrapper) -> T:  # noqa: C901,PLR0911
    # function, generator, coroutine & async generator
    # are distinguishable only by their result
    if isinstance(r, _Proxy):
        return r

    if isinstance(r, StopIteration) and _OP_FORK_STOPITER:
        r.value = _wrap(r.value, wrapper)
        return r

    # __await__, __iter__, __next__, send, throw, close
    if isinstance(r, Coroutine) and isinstance(r, Generator):
        return _CoroutineGenerator(r, wrapper)

    match r:
        # __aiter__, __anext__, asend, athrow, aclose
        case types.AsyncGeneratorType() if _OP_FUNC:  # asyncgen functions
            return _asyncgen(r, wrapper)  # type: ignore[return-value]
        case AsyncGenerator():  # user's asyncgens
            return _AsyncGenerator(r, wrapper)  # type: ignore[return-value]

        # __aiter__, __anext__
        case AsyncIterator():  # user's asynciters
            return _AsyncIterator(r, wrapper)  # type: ignore[return-value]

        # __await__, send, throw, close
        case types.CoroutineType() if _OP_FUNC:  # coroutine functions
            cr = _coroutine(r, wrapper)
            weakref.finalize(cr, r.close)
            return cr  # type: ignore[return-value]
        case Coroutine():  # user's coroutines
            return _Coroutine(r, wrapper)  # type: ignore[return-value]

        # __await__
        case Awaitable():  # Future-like from function
            return _FutureLike(r, wrapper)  # type: ignore[return-value]

        # __iter__, __next__, send, throw, close
        case types.GeneratorType() if _OP_FUNC:  # genfuncs
            return _gen(r, wrapper)  # type: ignore[return-value]
        case Generator():  # user's generators
            return _Generator(r, wrapper)  # type: ignore[return-value]

        # __iter__, __next__
        case Iterator():
            return _Iterator(r, wrapper)  # type: ignore[return-value]

    return r
