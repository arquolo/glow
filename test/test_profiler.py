import asyncio
import contextlib
import gc
import inspect
import re
import sys
import traceback
import warnings
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Coroutine,
    Generator,
    Iterator,
)
from enum import Enum
from types import (
    AsyncGeneratorType,
    CodeType,
    CoroutineType,
    FrameType,
    GeneratorType,
    coroutine,
)
from typing import Any, Literal, Never, Protocol, Self, cast, overload

import pytest

from glow import time_this

decorate = time_this()
# decorate = time_this(disable=True)


class MyError(Exception):
    pass


class BadBaseError(Exception):
    def __new__(cls, *args, **kwargs) -> type[Self]:  # type: ignore[misc]
        return cls


def gc_collect() -> None:
    gc.collect()
    gc.collect()
    gc.collect()


def throw(exc: type[Exception]) -> Never:
    raise exc


class _Empty(Enum):
    token = 0


class _BadTarget:
    def __setitem__(self, key, value) -> Never:
        raise StopAsyncIteration(42)


class _BadIterable:
    def __iter__(self) -> Iterator[Never]:
        raise StopAsyncIteration(42)


# ---------------------------- generator mixtures ----------------------------


def as_gen[Y, S, R](obj: Generator[Y, S, R]) -> GeneratorType[Y, S, R]:
    return cast('GeneratorType[Y, S, R]', obj)


@overload
def _gen_once() -> Generator[None, Any]: ...
@overload
def _gen_once[T](v: T, /) -> Generator[T, Any]: ...
@decorate
def _gen_once[T](v: T | None = None, /) -> Generator[T | None, Any]:
    yield v


@decorate
def gen_return_42() -> Generator[Never, Any, Literal[42]]:
    return 42
    yield


@decorate
def gen_yield_raises[T](
    value: T, exc_tp: type[Exception] = ValueError
) -> Generator[T, Any, Never]:
    yield value
    raise exc_tp


@decorate
def gen_raises(exc_tp: type[Exception]) -> Generator[Never, Any, Never]:
    raise exc_tp
    yield


@decorate
def gen_catching_genexit() -> Generator[None, Any, Literal[0] | None]:
    try:
        yield
    except GeneratorExit:
        return 0
    else:
        return None


@decorate
def gen_returning_send() -> Generator[Literal[1], Any, Any]:
    return (yield 1)


@decorate
def _gen_copy[T](*values: T) -> Generator[T]:
    for x in values:  # noqa: UP028
        yield x


def _sync_iterate[T](g: Iterator[T]) -> Generator[T | str]:
    while True:
        try:
            yield g.__next__()
        except StopIteration:
            yield 'STOP'
            return
        except Exception as ex:  # noqa: BLE001
            yield str(type(ex))


# ---------------------------- coroutine mixtures ----------------------------


def as_coro[Y, S, R](obj: Coroutine[Y, S, R]) -> CoroutineType[Y, S, R]:
    return cast('CoroutineType[Y, S, R]', obj)


@overload
def _cr_return() -> Coroutine[Any, Any, None]: ...
@overload
def _cr_return[T](v: T, /) -> Coroutine[Any, Any, T]: ...
@decorate
async def _cr_return[T](v: T | None = None, /) -> T | None:
    return v


@decorate
async def _cr_throw_stop() -> Never:
    raise StopIteration


class AwaitableGenerator[Y, S, R](Awaitable[R], Generator[Y, S, R]):
    pass


@overload
def _old_await() -> AwaitableGenerator[None, Any, None]: ...
@overload
def _old_await[Y](y: Y, /) -> AwaitableGenerator[Y, Any, None]: ...
@coroutine  # type: ignore[misc]
def _old_await[Y](y: Y | None = None, /) -> Generator[Y | None, Any]:
    yield y


@coroutine
def _old_suspend_return[Y, S](v: Y) -> Generator[Y, S, S]:
    send = yield v
    return send


@coroutine
def _old_suspend_suspend_return[S](v: int) -> Generator[int, S, S]:
    yield v * 10
    send = yield v * 10 + 1
    return send


class TestCoroType:
    @pytest.mark.asyncio
    async def test_cr_frame_f_back(self) -> None:
        cr = as_coro(_cr_return())
        assert cr.cr_frame
        assert cr.cr_frame.f_back is None
        await cr


class CatchUnraisableException:
    def __init__(self) -> None:
        self.unraisable = None
        self._old_hook = None

    def _hook(self, unraisable) -> None:
        self.unraisable = unraisable

    def __enter__(self) -> Self:
        self._old_hook = sys.unraisablehook
        sys.unraisablehook = self._hook
        return self

    def __exit__(self, *exc_info) -> None:
        sys.unraisablehook = self._old_hook
        del self.unraisable


class _AsyncYieldFrom[T]:
    def __init__(self, *values: T) -> None:
        self.values = values

    def __await__(self) -> Generator[T, Any]:
        yield from self.values


def _run_async[Y, R](
    obj: Awaitable[R] | Coroutine[Y, Any, R] | Generator[Y, Any, R],
) -> tuple[list[Y], R | None]:
    g = obj.__await__() if isinstance(obj, Awaitable) else obj

    buffer: list[Y] = []
    try:
        while True:
            buffer.append(g.send(None))
    except StopIteration as ex:
        return buffer, ex.value


@contextlib.contextmanager
def _silence_warnings(action: Literal['ignore', 'error']) -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter(action)
        yield


# ------------------------- async generator mixtures -------------------------


def as_agen[Y, S](obj: AsyncGenerator[Y, S]) -> AsyncGeneratorType[Y, S]:
    return cast('AsyncGeneratorType[Y, S]', obj)


@decorate
async def _agen_inf_aw_exc() -> AsyncGenerator[Never, Any]:
    while True:
        try:
            await _old_suspend_return(None)
        except MyError:
            pass
    return
    yield


@decorate
async def _agen_yi_inf_aw_exc() -> AsyncGenerator[None, Any]:
    try:
        yield
    except MyError:
        pass
    while True:
        try:
            await _old_suspend_return(None)
        except MyError:
            pass


@decorate
async def _agen_aw_exc_aw(
    exc_type: type[BaseException], x1=None, x2=None
) -> AsyncGenerator[Never, Any]:
    try:
        await _old_suspend_return(x1)
    except exc_type:
        await _old_suspend_return(x2)
    return
    yield


@decorate
async def _agen_yi_raise[T](
    exc: Exception | type[Exception], *xs: T
) -> AsyncGenerator[T, Any]:
    for x in xs:
        yield x
    raise exc


@decorate
async def old_agenfn_yi_aw_aw() -> AsyncGenerator[None, Any]:
    try:
        yield
    except MyError:
        try:
            await _old_suspend_return(None)
        except GeneratorExit:
            await _old_suspend_return(None)


@decorate
async def old_agenfn_inf_aw() -> AsyncGenerator[Never, Any]:
    while True:
        await _old_suspend_return(1)
    return
    yield


@overload
def _agen_copy() -> AsyncGenerator[Never, Any]: ...
@overload
def _agen_copy[T](*values: T) -> AsyncGenerator[T, Any]: ...
@decorate
async def _agen_copy[T](*values: T) -> AsyncGenerator[T, Any]:
    for x in values:
        yield x


class _AsyncIterable:
    async def __aiter__(self) -> AsyncIterator[Literal[1, 2]]:
        yield 1
        yield 2


class _AsyncIterator:
    def __init__(self) -> None:
        self.i = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> int:
        if self.i > 10:
            raise StopAsyncIteration
        self.i += 1
        return self.i


@decorate
async def _badpairs() -> AsyncGenerator[_BadIterable, Any]:
    yield decorate(_BadIterable)()


@decorate
async def _async_gen_asyncio_anext() -> AsyncGenerator[
    Literal[1, 2, 3, 4, 1000], Any
]:
    yield 1
    await asyncio.sleep(0.01)
    try:
        yield 2
        yield 3
    except ZeroDivisionError:
        yield 1000
    await asyncio.sleep(0.01)
    yield 4


def _to_list[T](ait: AsyncIterator[T]) -> list[T]:
    @decorate
    async def iterate() -> list[T]:
        return [x async for x in ait]

    cr = iterate()
    exc = False
    while True:
        try:
            fut = cr.throw(MyError) if exc else cr.send(None)
        except StopIteration as ex:
            return ex.value
        exc = fut == ('throw',)


@overload
def py_anext[T](iterator: AsyncIterator[T]) -> Awaitable[T]: ...
@overload
def py_anext[T, T2](
    iterator: AsyncIterator[T], default: T2, /
) -> Awaitable[T | T2]: ...


def py_anext[T, T2](
    iterator: AsyncIterator[T], default: T2 | _Empty = _Empty.token
) -> Awaitable[T | T2]:
    """Pure-Python implementation of anext() for testing purposes.

    Closely matches the builtin anext() C implementation.
    Can be used to compare the built-in implementation of the inner
    coroutines machinery to C-implementation of __anext__() and send()
    or throw() on the returned generator.
    """

    try:
        __anext__ = type(iterator).__anext__
    except AttributeError as exc:
        raise TypeError(f'{iterator!r} is not an async iterator') from exc

    if default is _Empty.token:
        return __anext__(iterator)

    @decorate  # ! FIXME smh coroutine breaks drops to 202/203
    async def anext_impl() -> T | T2:
        # The C code is way more low-level than this, as it implements
        # all methods of the iterator protocol. In this implementation
        # we're relying on higher-level coroutine concepts, but that's
        # exactly what we want -- crosstest pure-Python high-level
        # implementation and low-level C anext() iterators.
        aw = __anext__(iterator)
        try:
            ret = await aw
        except StopAsyncIteration:
            return default
        else:
            return ret

    return anext_impl()


class Anext(Protocol):
    @overload
    async def __call__[T](self, x: AsyncIterator[T], /) -> T: ...

    @overload
    async def __call__[T, T2](
        self, x: AsyncIterator[T], default: T2, /
    ) -> T | T2: ...


def _async_iterate[T](g: AsyncIterator[T]) -> Generator[T | str]:
    an = g.__anext__().__await__()
    while True:
        try:
            an.__next__()

        except StopAsyncIteration:
            yield 'STOP'
            return

        except StopIteration as ex:
            try:
                yield 'EMPTY StopIteration' if ex.value is None else ex.value
            except StopAsyncIteration:
                yield 'STOP'
                return
            an = g.__anext__().__await__()

        except Exception as ex:  # noqa: BLE001
            try:
                yield str(type(ex))
            except StopAsyncIteration:
                yield 'STOP'
                return
            an = g.__anext__().__await__()


def gencmp[T](it: Iterator[T], ait: AsyncIterator[T]) -> list[T | str]:
    sync_gen_result = [*_sync_iterate(it)]
    async_gen_result = [*_async_iterate(ait)]
    assert sync_gen_result == async_gen_result
    return async_gen_result


# -------------------------------- generators --------------------------------


class TestGen:
    def test_send_non_none_to_new_gen(self) -> None:
        g = _gen_once(1)
        with pytest.raises(TypeError):
            g.send(0)
        assert next(g) == 1

    def test_empty(self) -> None:
        g = gen_return_42()
        with pytest.raises(StopIteration, check=lambda e: e.value == 42):
            next(g)

    def test_handle_frame_object_in_creation(self) -> None:
        # Attempt to expose partially constructed frames
        # See https://github.com/python/cpython/issues/94262

        thresholds = gc.get_threshold()
        gc.callbacks.append(lambda *args: inspect.stack())
        gc.set_threshold(1, 0, 0)
        try:
            _gen_once(1)
        finally:
            gc.set_threshold(*thresholds)
            gc.callbacks.pop()

        class Sneaky:
            _s: 'Sneaky | None'

            def __del__(self) -> None:
                inspect.stack()

        sneaky = Sneaky()
        sneaky._s = Sneaky()
        sneaky._s._s = sneaky

        gc.set_threshold(1, 0, 0)
        try:
            del sneaky
            _gen_once(1)
        finally:
            gc.set_threshold(*thresholds)

    def test_gi_frame_f_back(self) -> None:
        gi = as_gen(_gen_once())
        assert gi.gi_frame
        assert gi.gi_frame.f_back is None

    def test_issue103488(self) -> None:
        # This should not raise
        try:
            for _ in gen_yield_raises(None):
                pass
        except ValueError:
            pass

    def test_genexpr_only_calls_dunder_iter_once(self) -> None:
        @decorate
        class _Iterator:
            def __init__(self) -> None:
                self.val = 0

            def __next__(self) -> int:
                if self.val == 2:
                    raise StopIteration
                self.val += 1
                return self.val

            # No __iter__ method

        class C:
            def __iter__(self) -> _Iterator:
                return _Iterator()

        assert list(decorate(C)()) == [1, 2]

    def test_send_01(self) -> None:
        @decorate
        def gen() -> Generator[Any | Literal[1], Any]:
            v = yield 1
            yield v * 2

        g = gen()
        assert g.send(None) == 1
        assert g.send(100) == 200

    def test_next_01(self) -> None:
        @decorate
        def foo() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass

        g = foo()
        g.send(None)
        with pytest.raises(StopIteration):
            g.send(None)


# Tests for the issue #23353: check that the currently handled exception
# is correctly saved/restored in PyEval_EvalFrameEx().


class TestGenExceptions:
    def test_throw_1(self) -> None:
        @decorate
        def store_raise_exc_generator() -> Generator[None]:
            try:
                assert sys.exception() is None
                yield
            except Exception as exc:
                # exception raised by gen.throw(exc)
                assert isinstance(sys.exception(), ValueError)
                assert exc.__context__ is None  # noqa: PT017
                yield

                # ensure that the exception is not lost
                assert isinstance(sys.exception(), ValueError)
                yield

                # we should be able to raise back the ValueError
                raise

        make = store_raise_exc_generator()
        next(make)

        try:
            raise ValueError  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            try:
                make.throw(exc)
            except Exception:  # noqa: BLE001
                pass

        next(make)
        with pytest.raises(ValueError, check=lambda e: e.__context__ is None):
            next(make)

        assert sys.exception() is None

    def test_throw_2(self) -> None:
        @decorate
        def gen() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass

        g = gen()
        g.send(None)
        with pytest.raises(StopIteration):
            g.throw(ValueError)

    def test_next(self) -> None:
        @decorate
        def gen() -> Generator[Literal['done']]:
            assert isinstance(sys.exception(), ValueError)
            yield 'done'

        g = gen()
        try:
            raise ValueError  # noqa: TRY301
        except Exception:  # noqa: BLE001
            assert next(g) == 'done'
        assert sys.exception() is None

    def test_gen_except(self) -> None:
        @decorate
        def gen() -> Generator[Literal['done'] | None]:
            assert sys.exception() is None
            try:
                yield None
                # we are called from "except ValueError:", TypeError must
                # inherit ValueError in its context
                raise TypeError  # noqa: TRY301
            except TypeError as exc:
                assert isinstance(sys.exception(), TypeError)
                assert isinstance(exc.__context__, ValueError)  # noqa: PT017
            # here we are still called from the "except ValueError:"
            assert isinstance(sys.exception(), ValueError)
            yield None
            assert sys.exception() is None
            yield 'done'

        g = gen()
        next(g)
        try:
            raise ValueError  # noqa: TRY301
        except Exception:  # noqa: BLE001
            next(g)

        assert next(g) == 'done'
        assert sys.exception() is None

    def test_nested_gen_except_loop(self) -> None:
        @decorate
        def gen() -> Generator[Literal['doing']]:
            for _ in range(100):
                assert isinstance(sys.exception(), TypeError)
                yield 'doing'

        @decorate
        def outer() -> Generator[Literal['doing']]:
            try:
                raise TypeError
            except:  # noqa: E722
                yield from gen()

        try:
            raise ValueError  # noqa: TRY301
        except Exception:  # noqa: BLE001
            for x in outer():
                assert x == 'doing'
        assert sys.exception() is None

    def test_throw_exception_context(self) -> None:
        @decorate
        def gen() -> Generator[Literal['done'] | None]:
            try:
                try:
                    assert sys.exception() is None
                    yield None
                except ValueError:
                    # we are called from "except ValueError:"
                    assert isinstance(sys.exception(), ValueError)
                    raise TypeError  # noqa: B904
            except Exception as exc:  # noqa: BLE001
                assert isinstance(sys.exception(), TypeError)
                assert isinstance(exc.__context__, ValueError)  # noqa: PT017

            # we are still called from "except ValueError:"
            assert isinstance(sys.exception(), ValueError)
            yield None
            assert sys.exception() is None
            yield 'done'

        g = gen()
        next(g)
        try:
            raise ValueError  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            g.throw(exc)

        assert next(g) == 'done'
        assert sys.exception() is None

    def test_throw_bad_exception_1(self) -> None:
        g = _gen_once()
        with pytest.raises(
            TypeError,
            match='should have returned an instance of BaseException',
        ):
            g.throw(BadBaseError)

        with pytest.raises(StopIteration):
            next(g)

    def test_throw_bad_exception_2(self) -> None:
        @decorate
        def gen() -> Generator[None]:
            with pytest.raises(
                TypeError,
                match='should have returned an instance of BaseException',
            ):
                yield

        g = gen()
        next(g)
        with pytest.raises(StopIteration):
            g.throw(BadBaseError)

    def test_stopiteration_error(self) -> None:
        # See also PEP 479.
        with pytest.raises(RuntimeError, match='raised StopIteration'):
            next(gen_raises(StopIteration))

    def test_tutorial_stopiteration(self) -> None:
        # Raise StopIteration" stops the generator too:

        g = gen_yield_raises(1, StopIteration)
        assert next(g) == 1

        with pytest.raises(RuntimeError, match='raised StopIteration'):
            next(g)

    def test_return_tuple(self) -> None:
        g = gen_returning_send()
        assert next(g) == 1
        with pytest.raises(StopIteration, check=lambda e: e.value == (2,)):
            g.send((2,))

    def test_return_stopiteration(self) -> None:
        g = gen_returning_send()
        assert next(g) == 1
        with pytest.raises(
            StopIteration,
            check=lambda e: (
                isinstance(e.value, StopIteration) and e.value.value == 2
            ),
        ):
            g.send(StopIteration(2))


class TestGenClose:
    def test_no_return_value(self) -> None:
        g = _gen_once()
        g.send(None)
        assert g.close() is None

    def test_no_except(self) -> None:
        def foo() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass

        g = foo()
        g.send(None)
        g.close()

    def test_return_value(self) -> None:
        @decorate
        def gen() -> Generator[None, Any, Literal[0] | None]:
            try:
                yield
                # close() raises GeneratorExit here, which is caught
            except GeneratorExit:
                return 0
            else:
                return None

        g = gen()
        g.send(None)
        assert g.close() == 0

    def test_not_catching_exit(self) -> None:
        @decorate
        def gen() -> Generator[None, Any, Literal[0]]:
            yield
            # close() raises GeneratorExit here, which isn't caught and
            # therefore propagates -- no return value
            return 0

        g = gen()
        g.send(None)
        assert g.close() is None

    def test_not_started(self) -> None:
        g = gen_catching_genexit()
        assert g.close() is None

    def test_exhausted(self) -> None:
        g = gen_catching_genexit()
        next(g)
        with pytest.raises(StopIteration):
            next(g)
        assert g.close() is None

    def test_closed(self) -> None:
        g = gen_catching_genexit()
        g.send(None)
        assert g.close() == 0
        assert g.close() is None

    def test_raises(self) -> None:
        @decorate
        def gen() -> Generator[None, Any, Never]:
            try:
                yield
            except GeneratorExit:
                pass
            raise RuntimeError

        g = gen()
        g.send(None)
        with pytest.raises(RuntimeError):
            g.close()

    def test_releases_frame_locals(self) -> None:
        # See gh-118272

        class Foo:
            pass

        f = Foo()
        f_wr = weakref.ref(f)

        @decorate
        def gen() -> Generator[None]:
            a = f  # noqa: F821, F841
            yield

        g = gen()
        next(g)
        del f
        g.close()
        gc_collect()
        assert f_wr() is None

    def test_gen_close_11(self) -> None:
        @decorate
        def gen() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass
            yield

        g = gen()
        g.send(None)
        with pytest.raises(RuntimeError, match='ignored GeneratorExit'):
            g.close()


class TestGenThrow:
    def test_01(self) -> None:
        @decorate
        def gen() -> Generator[Any | int, Any]:
            try:
                v = yield 1
            except MyError:
                yield 2000
            else:
                yield v * 2

        g = gen()
        assert g.send(None) == 1
        assert g.throw(MyError) == 2000
        with pytest.raises(StopIteration):
            g.send(None)

    def test_exception_context_with_yield(self) -> None:
        @decorate
        def gen() -> Generator[None]:
            try:
                raise KeyError('a')  # noqa: TRY301
            except Exception:  # noqa: BLE001
                yield

        g = gen()
        g.send(None)
        with pytest.raises(
            ValueError,
            check=lambda e: (
                isinstance(e.__context__, KeyError)
                and e.__context__.args == ('a',)
            ),
        ):
            g.throw(ValueError)

    def test_exception_context_with_yield_inside_generator(self) -> None:
        # Check that the context is also available from inside the generator
        # with yield, as opposed to outside.
        @decorate
        def gen() -> Generator[None | Literal['b']]:
            try:
                raise KeyError('a')  # noqa: TRY301
            except Exception:  # noqa: BLE001
                try:
                    yield None
                except Exception as exc:  # noqa: BLE001
                    assert isinstance(exc, ValueError)  # noqa: PT017
                    context = exc.__context__
                    assert isinstance(context, KeyError)
                    assert context.args == ('a',)
                    yield 'b'

        g = gen()
        g.send(None)
        assert g.throw(ValueError) == 'b'

    def test_exception_context_with_yield_from(self) -> None:
        @decorate
        def gen() -> Generator[None]:
            try:
                raise KeyError('a')  # noqa: TRY301
            except Exception:  # noqa: BLE001
                yield from _gen_once()

        g = gen()
        g.send(None)
        with pytest.raises(
            ValueError,
            check=lambda e: (
                isinstance(e.__context__, KeyError)
                and e.__context__.args == ('a',)
            ),
        ):
            g.throw(ValueError)

    def test_exception_context_with_yield_from_with_cycle(self) -> None:
        # Check trying to create an exception context cycle:
        # https://bugs.python.org/issue40696
        has_cycle = None

        @decorate
        def gen(exc) -> Generator[None]:
            nonlocal has_cycle
            try:
                raise exc  # noqa: TRY301
            except Exception:  # noqa: BLE001
                try:
                    yield from _gen_once()
                except Exception as exc_:  # noqa: BLE001
                    has_cycle = exc_ is exc_.__context__
            yield

        exc = KeyError('a')
        g = gen(exc)
        g.send(None)
        g.throw(exc)
        # This also distinguishes from the initial has_cycle=None.
        assert has_cycle is False

    def test_throw_after_none_exc_type(self) -> None:
        @decorate
        def gen() -> Generator[None]:
            try:
                raise KeyError  # noqa: TRY301
            except KeyError:
                pass

            try:
                yield
            except Exception:  # noqa: BLE001
                raise RuntimeError  # noqa: B904

        g = gen()
        g.send(None)
        with pytest.raises(RuntimeError):
            g.throw(ValueError)


class TestGenPep479:
    def test_stopiteration_wrapping(self) -> None:
        @decorate
        def gen() -> Generator[Never, Any, Never]:
            yield throw(StopIteration)

        with pytest.raises(
            RuntimeError, match='generator raised StopIteration'
        ):
            next(gen())

    def test_stopiteration_wrapping_context(self) -> None:
        @decorate
        def gen() -> Generator[Never, Any, Never]:
            yield throw(StopIteration)

        with pytest.raises(
            RuntimeError,
            check=lambda exc: (
                isinstance(exc.__cause__, StopIteration)
                and isinstance(exc.__context__, StopIteration)
                and exc.__suppress_context__
            ),
        ):
            next(gen())


# -------------------------------- coroutines --------------------------------


class TestCoroutine:
    def test_gen_1(self) -> None:
        assert not hasattr(_gen_once, '__await__')

    def test_func_1(self) -> None:
        cr = _cr_return(10)
        assert isinstance(cr, CoroutineType)
        assert _cr_return.__code__.co_flags & inspect.CO_COROUTINE
        assert not (_cr_return.__code__.co_flags & inspect.CO_GENERATOR)
        assert cr.cr_code.co_flags & inspect.CO_COROUTINE
        assert not (cr.cr_code.co_flags & inspect.CO_GENERATOR)
        assert _run_async(cr) == ([], 10)
        assert _run_async(_cr_return(10)) == ([], 10)

        @decorate
        def bar() -> None:
            pass

        assert not (bar.__code__.co_flags & inspect.CO_COROUTINE)

    def test_func_2(self) -> None:
        with pytest.raises(
            RuntimeError, match='coroutine raised StopIteration'
        ):
            _run_async(_cr_throw_stop())

    def test_func_3(self) -> None:
        cr = _cr_throw_stop()
        assert re.match('^<coroutine object.* at 0x.*>$', repr(cr))
        cr.close()

    def test_func_4(self) -> None:
        for el in _old_await(1):
            assert el == 1
        assert list(_old_await(1)) == [1]
        assert tuple(_old_await(1)) == (1,)
        assert next(iter(_old_await(1))) == 1

    def test_func_5(self) -> None:
        @coroutine
        def bar() -> Generator[Literal[1, 2]]:
            yield 1
            yield 2

        async def foo() -> None:
            await bar()

        cr = foo()
        assert cr.send(None) == 1
        assert cr.send(None) == 2
        with pytest.raises(StopIteration):
            cr.send(None)

    def test_func_6(self) -> None:
        @coroutine
        def bar() -> Generator[Any, Any, str]:
            return (yield from cr)

        cr = _cr_return('spam')
        assert _run_async(bar()) == ([], 'spam')
        cr.close()

    def test_func_7(self) -> None:
        with pytest.warns(  # noqa: PT031
            RuntimeWarning, match=r"coroutine '.*' was never awaited"
        ):
            _cr_return()
            gc_collect()

        with pytest.warns(  # noqa: PT031
            RuntimeWarning, match=r"coroutine '.*' was never awaited"
        ):
            with pytest.raises(TypeError):  # See bpo-32703.
                for _ in _cr_return():
                    pass
            gc_collect()

    def test_func_8(self) -> None:
        n = 0

        @coroutine
        def gen() -> Generator[int | None, int]:
            nonlocal n
            try:
                a = yield None
                yield (a**2)
            except ZeroDivisionError:
                n += 100
                raise
            finally:
                n += 1

        @decorate
        async def foo() -> None:
            await gen()

        cr = foo()
        g = cr.__await__()
        assert g is iter(g)
        next(g)
        assert g.send(10) == 100

        assert n == 0
        g.close()
        assert n == 1

        cr = foo()
        g = cr.__await__()
        next(g)
        with pytest.raises(ZeroDivisionError):
            g.throw(ZeroDivisionError())
        assert n == 102

        cr = foo()
        g = cr.__await__()
        next(g)
        with (
            pytest.raises(ZeroDivisionError),
            pytest.warns(DeprecationWarning),  # noqa: PT030
        ):
            g.throw(ZeroDivisionError, ZeroDivisionError(), None)

    def test_func_9(self) -> None:
        cr = _cr_return()
        # Test that PyCoro_Type and _PyCoroWrapper_Type types were properly
        # initialized
        assert '__await__' in dir(cr)
        assert '__iter__' in dir(cr.__await__())
        assert 'coroutine_wrapper' in repr(cr.__await__())
        cr.close()  # avoid RuntimeWarning

    def test_func_10(self) -> None:
        async def coro():
            assert cr is not None
            cr.send(None)
            await asyncio.sleep(0)

        cr = coro()
        with pytest.raises(ValueError, match='coroutine already executing'):
            cr.send(None)

    def test_func_11(self) -> None:
        cr = _cr_return()
        with pytest.raises(
            TypeError,
            match="can't send non-None value to a just-started coroutine",
        ):
            cr.send('spam')

        cr.close()

    def test_func_12(self) -> None:
        @decorate
        async def coro() -> None:
            try:
                await _old_await()
            except GeneratorExit:
                await _old_await()

        cr = coro()
        cr.send(None)
        with pytest.raises(
            RuntimeError, match='coroutine ignored GeneratorExit'
        ):
            cr.close()

    def test_func_13(self) -> None:
        # See http://bugs.python.org/issue25887 for details

        @decorate
        async def reader[T](coro: Awaitable[T]) -> T:
            return await coro

        cr_spam = _cr_return('spam')

        with pytest.raises(StopIteration, match='spam'):
            reader(cr_spam).send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            reader(cr_spam).send(None)

    def test_func_14(self) -> None:
        # See http://bugs.python.org/issue25887 for details
        @decorate
        async def send() -> Literal['spam']:
            await _old_await()
            return 'spam'

        @decorate
        async def read[T](coro: Awaitable[T]) -> T:
            await _old_await()
            return await coro

        cr_spam = send()

        cr_read = read(cr_spam)
        cr_read.send(None)
        cr_read.send(None)
        with pytest.raises(Exception, match='ham'):
            cr_read.throw(Exception('ham'))

        cr_read = read(cr_spam)
        cr_read.send(None)
        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            cr_read.send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            cr_read.throw(Exception('wat'))

    def test_func_15(self) -> None:
        # See http://bugs.python.org/issue25887 for details

        cr = _cr_return('spam')
        with pytest.raises(StopIteration, match='spam'):
            cr.send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            cr.send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            cr.throw(Exception('wat'))

        # Closing a coroutine shouldn't raise any exception even if it's
        # already closed/exhausted (similar to generators)
        cr.close()
        cr.close()

    def test_func_16(self) -> None:
        # See http://bugs.python.org/issue25887 for details
        cr = _cr_return('spam')
        await_iter = cr.__await__()
        it = iter(await_iter)

        with pytest.raises(StopIteration, match='spam'):
            it.send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            it.send(None)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            # Although the iterator protocol requires iterators to
            # raise another StopIteration here, we don't want to do
            # that.  In this particular case, the iterator will raise
            # a RuntimeError, so that 'yield from' and 'await'
            # expressions will trigger the error, instead of silently
            # ignoring the call.
            next(it)

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            it.throw(Exception('wat'))

        with pytest.raises(
            RuntimeError, match='cannot reuse already awaited coroutine'
        ):
            it.throw(Exception('wat'))

        # Closing a coroutine shouldn't raise any exception even if it's
        # already closed/exhausted (similar to generators)
        it.close()
        it.close()

    def test_func_17(self) -> None:
        check = 0

        @coroutine
        def coro1() -> Generator[None]:
            nonlocal check
            yield
            try:
                yield
            except GeneratorExit:
                check += 1

        @decorate
        async def coro2() -> None:
            await coro1()

        cr = coro2()
        cr.send(None)
        cr.send(None)
        assert check == 0
        cr.close()
        assert check == 1

        for _ in range(3):
            # Closing a coroutine shouldn't raise any exception even if it's
            # already closed/exhausted (similar to generators)
            cr.close()
            assert check == 1

    def test_coro_wrapper_send_tuple(self) -> None:
        result = _run_async(_cr_return((10,)))
        assert result == ([], (10,))

    def test_coro_wrapper_send_stop_iterator(self) -> None:
        result = _run_async(_cr_return(StopIteration(10)))
        assert isinstance(result[1], StopIteration)
        assert result[1].value == 10

    def test_cr_await(self) -> None:
        @coroutine
        def coro1() -> Generator[None, Any]:
            assert inspect.getcoroutinestate(cr) == inspect.CORO_RUNNING
            assert cr.cr_await is None
            yield
            assert inspect.getcoroutinestate(cr) == inspect.CORO_RUNNING
            assert cr.cr_await is None

        @decorate
        async def coro2() -> None:
            await coro1()

        @decorate
        async def coro3() -> None:
            assert cr.cr_await is None
            await coro2()
            assert cr.cr_await is None

        cr = coro3()
        assert inspect.getcoroutinestate(cr) == inspect.CORO_CREATED
        assert cr.cr_await is None

        cr.send(None)
        assert inspect.getcoroutinestate(cr) == inspect.CORO_SUSPENDED
        assert cr.cr_await is not None

        with pytest.raises(StopIteration):
            cr.send(None)  # complete coroutine
        assert inspect.getcoroutinestate(cr) == inspect.CORO_CLOSED
        assert cr.cr_await is None

    def test_corotype_1(self) -> None:
        ct = CoroutineType
        assert ct.__name__ == 'coroutine'

        cr = _cr_return()
        assert 'coroutine object' in repr(cr)
        cr.close()

    def test_await_1(self) -> None:
        @decorate
        async def coro() -> None:
            await _AsyncYieldFrom(1, 2, 3)

        assert _run_async(coro()) == ([1, 2, 3], None)

    def test_await_2(self) -> None:
        @decorate
        async def coro() -> Literal[42]:
            return await _cr_return(42)

        assert _run_async(coro()) == ([], 42)

    def test_await_3(self) -> None:
        class _Awaitable:
            def __await__(self) -> Generator[int]:
                return (x for x in [52])

        @decorate
        async def foo():
            return await _Awaitable()

        assert _run_async(foo()) == ([52], None)

    def test_await_4(self) -> None:
        class Awaitable:
            def __await__(self) -> Generator[Literal[42], Any, Literal[100]]:
                yield 42
                return 100

        @decorate
        async def foo() -> Literal[100]:
            return await Awaitable()

        assert _run_async(foo()) == ([42], 100)

    def test_await_5(self) -> None:
        @decorate
        async def foo() -> int:
            return await _cr_return(42)

        @decorate
        async def foo2() -> int:
            return -await _cr_return(42)

        assert _run_async(foo()) == ([], 42)
        assert _run_async(foo2()) == ([], -42)

    def test_await_6(self) -> None:
        @decorate
        async def bar() -> Awaitable[Literal[42]]:
            return _cr_return(42)

        @decorate
        async def foo() -> Literal[42]:
            return await (await bar())

        assert _run_async(foo()) == ([], 42)

    def test_await_7(self) -> None:
        @decorate
        async def foo2() -> tuple[Literal['spam'], Literal['ham']]:
            return await _cr_return('spam'), 'ham'

        assert _run_async(foo2()) == ([], ('spam', 'ham'))

    def test_await_8(self) -> None:
        class FutureLike:
            def __await__(
                self,
            ) -> Generator[None, Literal['spam'], Literal['spam']]:
                return (yield)

        @decorate
        async def coro1() -> Literal['spam']:
            try:
                return await FutureLike()
            except ZeroDivisionError:
                raise MyError from None

        class Wrapper[R]:
            # Forces the interpreter to use CoroutineType.__await__
            def __init__(self, coro: Coroutine[Any, Any, R]) -> None:
                assert coro.__class__ is CoroutineType
                self.coro = coro

            def __await__(self) -> Generator[Any, Any, R]:
                return self.coro.__await__()

        @decorate
        async def coro2() -> Literal['spam']:
            return await Wrapper(coro1())

        cr = coro2()
        cr.send(None)
        with pytest.raises(StopIteration, match='spam'):
            cr.send('spam')

        cr = coro2()
        cr.send(None)
        with pytest.raises(MyError):
            cr.throw(ZeroDivisionError)

    def test_await_9(self) -> None:
        @decorate
        async def coro() -> None:
            await _old_await()

        @decorate
        async def waiter(coro) -> None:
            await coro

        cr = coro()
        cr.send(None)

        with pytest.raises(
            RuntimeError, match='coroutine is being awaited already'
        ):
            waiter(cr).send(None)

    def test_await_10(self) -> None:
        # See https://bugs.python.org/issue29600 for details.
        @decorate
        async def coro() -> ValueError:
            try:
                raise KeyError  # noqa: TRY301
            except KeyError:
                return await _cr_return(ValueError())

        _, result = _run_async(coro())
        assert result is not None
        assert result.__context__ is None

    def test_comp_1(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [await c for c in [_cr_return(1), _cr_return(41)]]

        assert _run_async(coro()) == ([], [1, 41])

    def test_comp_2(self) -> None:
        @decorate
        async def coro() -> list[str]:
            return [
                s
                for c in [
                    _cr_return(''),
                    _cr_return('abc'),
                    _cr_return(''),
                    _cr_return(['de', 'fg']),
                ]
                for s in await c
            ]

        assert _run_async(coro()) == ([], ['a', 'b', 'c', 'de', 'fg'])

    def test_comp_3(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [
                i + 1
                for pair in ([10, 20], [30, 40])
                if pair[0] > 10
                async for i in _agen_copy(*pair)
                if i > 30
            ]

        assert _run_async(coro()) == ([], [41])

    def test_comp_4(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [
                i + 1 async for seq in _agen_copy((10, 20), (30,)) for i in seq
            ]

        assert _run_async(coro()) == ([], [11, 21, 31])

    def test_comp_5(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [1, 2, 3]

        assert _run_async(coro()) == ([], [1, 2, 3])

    def test_fatal_coro_warning(self) -> None:
        # Issue 27811
        with (
            _silence_warnings('error'),
            CatchUnraisableException() as cm,
        ):
            # avoid keeping the coroutine alive
            _cr_return()
            gc_collect()

            assert re.match(
                r'Exception ignored while finalizing coroutine .*',
                cm.unraisable.err_msg,
            )
            assert 'was never awaited' in str(cm.unraisable.exc_value)

    def test_bpo_45813_1(self) -> None:
        """This would crash the interpreter in 3.11a2"""
        with pytest.warns(RuntimeWarning):
            frame = as_coro(_cr_return()).cr_frame
        assert frame is not None
        frame.clear()

    def test_bpo_45813_2(self) -> None:
        """This would crash the interpreter in 3.11a2"""
        cr = as_coro(_cr_return())
        assert cr.cr_frame is not None
        with pytest.warns(RuntimeWarning):
            cr.cr_frame.clear()
        cr.close()

    def test_cr_frame_after_close(self) -> None:
        cr = as_coro(_cr_return())
        assert cr.cr_frame is not None
        cr.close()
        assert cr.cr_frame is None

    def test_stack_in_coroutine_throw(self) -> None:
        # Regression test for https://github.com/python/cpython/issues/93592
        @coroutine
        def coro1() -> Generator[int, Any]:
            try:
                yield len(traceback.extract_stack())
            except ZeroDivisionError:
                yield len(traceback.extract_stack())

        @decorate
        async def coro2() -> None:
            return await coro1()

        @decorate
        async def coro3() -> None:
            return await coro2()

        cr = coro3()
        len_send = cr.send(None)
        len_throw = cr.throw(ZeroDivisionError)
        # before fixing, visible stack from throw would
        # be shorter than from send.
        assert len_send == len_throw

    def test_call_generator_in_frame_clear(self) -> None:
        # gh-143939: Running a generator while clearing the coroutine's frame
        # should not be misinterpreted as a yield.
        class CallGeneratorOnDealloc:
            def __del__(self) -> None:
                next(x for x in [1])

        @decorate
        async def coro() -> Literal[42]:
            obj = CallGeneratorOnDealloc()  # noqa: F841
            return 42

        yielded, result = _run_async(coro())
        assert yielded == []
        assert result == 42


# ----------------------------- async iteration ------------------------------


class TestAsyncFor:
    def test_stop_iteration(self) -> None:
        class AIter(StopIteration):
            i = 0

            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> int:
                if self.i:
                    raise StopAsyncIteration
                self.i += 1
                return self.value

        result = []

        @decorate
        async def foo() -> Never:
            async for i in decorate(AIter)(42):
                result.append(i)
            raise MyError

        with pytest.raises(MyError):
            foo().send(None)
        assert result == [42]

    def test_1(self) -> None:
        aiter_calls = 0

        class AsyncIter:
            def __init__(self) -> None:
                self.i = 0

            def __aiter__(self) -> Self:
                nonlocal aiter_calls
                aiter_calls += 1
                return self

            async def __anext__(self) -> tuple[int, int]:
                self.i += 1
                if not (self.i % 10):
                    await _AsyncYieldFrom(self.i * 10)
                if self.i > 100:
                    raise StopAsyncIteration
                return self.i, self.i

        buffer = []

        @decorate
        async def coro() -> None:
            async for i1, i2 in decorate(AsyncIter)():
                buffer.append(i1 + i2)

        yielded, _ = _run_async(coro())
        # Make sure that __aiter__ was called only once
        assert aiter_calls == 1
        assert yielded == [i * 100 for i in range(1, 11)]
        assert buffer == [i * 2 for i in range(1, 101)]

    def test_2(self) -> None:
        aiter_calls = 0

        class AsyncIter:
            def __init__(self) -> None:
                self.i = 0

            def __aiter__(self) -> Self:
                nonlocal aiter_calls
                aiter_calls += 1
                return self

            async def __anext__(self) -> tuple[int, int]:
                self.i += 1
                if not (self.i % 10):
                    await _AsyncYieldFrom(self.i * 10)
                if self.i > 100:
                    raise StopAsyncIteration
                return self.i, self.i

        buffer = []

        @decorate
        async def coro() -> None:
            nonlocal buffer
            async for i in decorate(AsyncIter)():
                if i[0] > 20:
                    continue
                buffer.append(i[0])
            buffer.append('what?')
            buffer.append('end')

        yielded, _ = _run_async(coro())
        # Make sure that __aiter__ was called only once
        assert aiter_calls == 1
        assert yielded == [i * 100 for i in range(1, 11)]
        assert buffer == [*range(1, 21), 'what?', 'end']

    def test_3(self) -> None:
        i = 0
        iterable = decorate(_AsyncIterator)()
        irefs_before = sys.getrefcount(iterable)

        @decorate
        async def coro() -> None:
            nonlocal i
            async for _ in iterable:
                i += 1
            i += 1000

        with _silence_warnings('error'):
            # Test that __aiter__ that returns an asynchronous iterator
            # directly does not throw any warnings.
            _run_async(coro())
        assert i == 1011

        assert sys.getrefcount(iterable) == irefs_before

    def test_4(self) -> None:
        i = 0

        @decorate
        async def coro() -> None:
            nonlocal i
            i += 100
            async for _ in decorate(_AsyncIterator)():
                i += 1
            i += 1000
            async for _ in decorate(_AsyncIterator)():
                i += 1
            i += 10000

        _run_async(coro())
        assert i == 11122

    def test_tuple(self) -> None:
        class AIter[T](tuple[T]):
            i = 0

            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> T:
                if self.i >= len(self):
                    raise StopAsyncIteration
                self.i += 1
                return self[self.i - 1]

        result = []

        @decorate
        async def foo() -> Never:
            async for i in decorate(AIter)([42]):
                result.append(i)
            raise MyError

        with pytest.raises(MyError):
            foo().send(None)
        assert result == [42]

    def test_comp_1(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [i + 1 async for i in _agen_copy(10, 20)]

        assert _run_async(coro()) == ([], [11, 21])

    def test_comp_2(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [i + 1 async for i in _agen_copy(10, 20) if i > 10]

        assert _run_async(coro()) == ([], [21])

    def test_comp_3(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [i + 10 async for i in _agen_copy(*range(5)) if 0 < i < 4]

        assert _run_async(coro()) == ([], [11, 12, 13])

    def test_comp_4(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [i async for i in _agen_yi_raise(MyError('aaa'), 1, 2)]

        with pytest.raises(MyError, match='aaa'):
            _run_async(coro())

    def test_comp_5(self) -> None:
        @decorate
        async def coro() -> list[int]:
            return [i async for i in _agen_copy(1, 2)]

        assert _run_async(coro()) == ([], [1, 2])

    def test_nested_cmp_list_in_list(self) -> None:
        @decorate
        async def coro() -> list[list[int]]:
            return [[i + j async for i in _agen_copy(1, 2)] for j in [10, 20]]

        assert _run_async(coro()) == ([], [[11, 12], [21, 22]])

    def test_nested_cmp_list_in_gen(self) -> None:
        @decorate
        async def coro() -> list[list[int]]:
            agen = ([i + j async for i in _agen_copy(1, 2)] for j in [10, 20])
            return [x async for x in agen]

        assert _run_async(coro()) == ([], [[11, 12], [21, 22]])

    def test_nested_cmp_gen_in_list(self) -> None:
        @decorate
        async def coro() -> list[int]:
            gens = [(i async for i in _agen_copy(*range(j))) for j in [3, 5]]
            return [x for g in gens async for x in g]

        assert _run_async(coro()) == (
            [],
            [0, 1, 2, 0, 1, 2, 3, 4],
        )

    def test_nested_cmp_gen_in_gen(self) -> None:
        @decorate
        async def coro() -> list[int]:
            gens = ((i async for i in _agen_copy(*range(j))) for j in [3, 5])
            return [x for g in gens async for x in g]

        assert _run_async(coro()) == (
            [],
            [0, 1, 2, 0, 1, 2, 3, 4],
        )

    def test_nested_cmp_list_in_list_in_list(self) -> None:
        @decorate
        async def coro() -> list[list[list[int]]]:
            return [
                [[i + j + k async for i in _agen_copy(1, 2)] for j in [10, 20]]
                for k in [100, 200]
            ]

        assert _run_async(coro()) == (
            [],
            [[[111, 112], [121, 122]], [[211, 212], [221, 222]]],
        )

    def test_assign_raising_stop_async_iter_1_for(self) -> None:
        tgt = _BadTarget()

        @decorate
        async def coro() -> Literal['end']:
            with pytest.raises(
                StopAsyncIteration, check=lambda e: e.args == (42,)
            ):
                async for tgt[0] in _agen_copy(10):
                    pass
            return 'end'

        assert _run_async(coro()) == ([], 'end')

    def test_assign_raising_stop_async_iter_1_list(self) -> None:
        tgt = _BadTarget()

        @decorate
        async def coro() -> list[int] | Literal['end']:
            with pytest.raises(
                StopAsyncIteration, check=lambda e: e.args == (42,)
            ):
                return [0 async for tgt[0] in _agen_copy(10)]
            return 'end'

        assert _run_async(coro()) == ([], 'end')

    def test_assign_raising_stop_async_iter_1_gen(self) -> None:
        tgt = _BadTarget()

        @decorate
        async def coro() -> Literal['end']:
            ag = (0 async for tgt[0] in _agen_copy(10))
            cr = ag.asend(None)
            with pytest.raises(
                RuntimeError,
                check=lambda e: (
                    isinstance(e.__cause__, StopAsyncIteration)
                    and e.__cause__.args == (42,)
                ),
            ):
                await cr
            return 'end'

        assert _run_async(coro()) == ([], 'end')

    def test_assign_raising_stop_async_iter_2_for(self) -> None:
        @decorate
        async def coro() -> Literal['end']:
            with pytest.raises(
                StopAsyncIteration, check=lambda e: e.args == (42,)
            ):
                async for _, _ in _badpairs():
                    pass
            return 'end'

        assert _run_async(coro()) == ([], 'end')

    def test_assign_raising_stop_async_iter_2_list(self) -> None:
        @decorate
        async def coro() -> Literal['end'] | list[int]:
            with pytest.raises(
                StopAsyncIteration, check=lambda e: e.args == (42,)
            ):
                return [0 async for _, _ in _badpairs()]
            return 'end'

        assert _run_async(coro()) == ([], 'end')

    def test_assign_raising_stop_async_iter_2_gen(self) -> None:
        @decorate
        async def coro() -> Literal['end']:
            ag = (0 async for _, _ in _badpairs())
            cr = ag.asend(None)
            with pytest.raises(
                RuntimeError,
                check=lambda e: (
                    isinstance(e.__cause__, StopAsyncIteration)
                    and e.__cause__.args == (42,)
                ),
            ):
                await cr
            return 'end'

        assert _run_async(coro()) == ([], 'end')


# ----------------------------- async generators -----------------------------


class TestAsyncGenType:
    def test_ag_frame_f_back(self) -> None:
        ag = as_agen(_agen_copy(None))
        assert ag.ag_frame
        assert ag.ag_frame.f_back is None


class TestAsyncGenIteration:
    def test_1(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123, 456, 789], Any]:
            await _old_await(('result',))
            s = yield 123
            assert s is None
            await _old_await(('result',))
            yield 456
            await _old_await(('result',))
            yield 789

        assert _to_list(agen()) == [123, 456, 789]

    def test_2(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123], Any]:
            await _old_await(('result',))
            yield 123
            await _old_await(('result',))

        ag = agen()
        ait = ag.__aiter__()

        aw = ait.__anext__()
        assert aw.__await__().__next__() == ('result',)

        with pytest.raises(StopIteration, check=lambda e: e.value == 123):
            aw.__await__().__next__()

        aw = ait.__anext__()
        assert aw.__await__().__next__() == ('result',)

        with pytest.raises(StopAsyncIteration, check=lambda e: not e.args):
            aw.__await__().__next__()


class TestAsyncGenException:
    def test_1(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123, 456], Any]:
            await _old_await(('result',))
            yield 123
            await _old_await(('throw',))
            yield 456

        with pytest.raises(MyError):
            _to_list(agen())

    def test_2(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123], Any]:
            await _old_await(('result',))
            yield 123
            raise ZeroDivisionError

        ag = agen()
        ait = ag.__aiter__()
        aw = ait.__anext__()
        assert aw.__await__().__next__() == ('result',)

        with pytest.raises(StopIteration, check=lambda e: e.value == 123):
            aw.__await__().__next__()

        with pytest.raises(ZeroDivisionError):
            ait.__anext__().__await__().__next__()

    def test_3(self) -> None:
        with pytest.raises(
            RuntimeError, match=r'async generator.*StopAsyncIteration'
        ):
            _to_list(_agen_yi_raise(StopAsyncIteration, 123))

    def test_4(self) -> None:
        with pytest.raises(
            RuntimeError, match=r'async generator.*StopIteration'
        ):
            _to_list(_agen_yi_raise(StopIteration, 123))

    def test_5(self) -> None:
        @decorate
        def sync_gen() -> Generator[Literal[1, 2, 3], Any, Never]:
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                yield 2
                yield 3

            yield 100

        @decorate
        async def async_gen() -> AsyncGenerator[Literal[1, 2, 3], Any]:
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                yield 2
                yield 3

            yield 100

        gencmp(sync_gen(), async_gen())

    def test_6(self) -> None:
        @decorate
        def sync_gen() -> Generator[Literal[1, 2], Any, Never]:
            try:
                yield 1
            finally:
                yield 2
                raise ZeroDivisionError
                yield 3

            yield 100

        @decorate
        async def async_gen() -> AsyncGenerator[Literal[1, 2], Any]:
            try:
                yield 1
                await _old_await(('result',))
            finally:
                await _old_await(('result',))
                yield 2
                raise ZeroDivisionError
                yield 3

            yield 100

        gencmp(sync_gen(), async_gen())

    def test_7(self) -> None:
        @decorate
        def sync_gen() -> Generator[Literal[1, 2, 3], Any, Never]:
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                yield 2
                yield 3

            yield 100

        @decorate
        async def async_gen() -> AsyncGenerator[Literal[1, 2, 3], Any]:
            try:
                await _old_await(('result',))
                yield 1
                raise ZeroDivisionError
            finally:
                yield 2
                await _old_await(('result',))
                yield 3

            yield 100

        gencmp(sync_gen(), async_gen())

    def test_8(self) -> None:
        with pytest.raises(
            TypeError, match=r'non-None value .* async generator'
        ):
            _agen_copy().__anext__().send(100)

    def test_9(self) -> None:
        @decorate
        def sync_gen() -> Iterator[Literal[10, 20, 30]]:
            yield 10
            g = _gen_copy(1, 2)
            g.send(None)
            try:
                g.throw(GeneratorExit())
            except GeneratorExit:
                yield 20
            yield 30

        @decorate
        async def async_gen() -> AsyncIterator[Literal[10, 20, 30]]:
            yield 10
            ag = _agen_copy(1, 2)
            await ag.asend(None)
            try:
                await ag.athrow(GeneratorExit())
            except GeneratorExit:
                yield 20
            yield 30

        gencmp(sync_gen(), async_gen())

    def test_10(self) -> None:
        @decorate
        async def agen() -> AsyncIterator[Literal[123]]:
            with pytest.raises(
                RuntimeError,
                match=r'anext\(\): asynchronous generator is already running',
            ):
                await anext(ag)
            yield 123

        ag = agen()
        ait = ag.__aiter__()
        aw = ait.__anext__()

        with pytest.raises(StopIteration):
            aw.__await__().__next__()

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            aw.__await__().send(None)

    @pytest.mark.asyncio
    async def test_11_async(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[None, Any]:
            nonlocal done
            try:
                yield
            except:  # noqa: E722
                pass
            done = 1

        ag = agen()
        await ag.asend(None)
        with pytest.raises(StopAsyncIteration):
            await ag.athrow(ValueError)
        assert done == 1


class TestOldAsyncGenAlreadyRunning:
    def test_asend_throw_concurrent_with_send(self) -> None:
        ag = _agen_inf_aw_exc()
        cr = ag.asend(None)
        cr.send(None)
        cr2 = ag.asend(None)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            cr2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            cr2.send(None)

    def test_athrow_throw_concurrent_with_send(self) -> None:
        ag = _agen_inf_aw_exc()
        cr = ag.asend(None)
        cr.send(None)
        cr2 = ag.athrow(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            cr2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            cr2.send(None)

    def test_asend_throw_concurrent_with_throw(self) -> None:
        ag = _agen_yi_inf_aw_exc()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)

        cr = ag.athrow(MyError)
        cr.throw(MyError)
        cr2 = ag.asend(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            cr2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            cr2.send(None)

    def test_athrow_throw_concurrent_with_throw(self) -> None:
        ag = _agen_yi_inf_aw_exc()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)

        cr = ag.athrow(MyError)
        cr.throw(MyError)

        cr2 = ag.athrow(MyError)
        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            cr2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            cr2.send(None)

    def test_asend_close_runtime_error(self) -> None:
        ag = _agen_aw_exc_aw(GeneratorExit)
        cr = ag.asend(None)
        cr.send(None)
        with pytest.raises(
            RuntimeError, match='coroutine ignored GeneratorExit'
        ):
            cr.close()

    def test_athrow_close_runtime_error(self) -> None:
        ag = old_agenfn_yi_aw_aw()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)
        cr = ag.athrow(MyError)
        cr.send(None)
        with pytest.raises(
            RuntimeError, match='coroutine ignored GeneratorExit'
        ):
            cr.close()


class TestAsyncGenApi:
    def test_frame(self) -> None:
        ag = as_agen(_agen_copy(1, 2))

        assert ag.ag_await is None
        assert isinstance(ag.ag_frame, FrameType)
        assert not ag.ag_running
        assert isinstance(ag.ag_code, CodeType)

        cr = ag.aclose()
        assert inspect.isawaitable(cr)
        cr.close()

    def test_aiter_idempotent(self) -> None:
        applied_once = aiter(_agen_copy())
        applied_twice = aiter(applied_once)
        assert applied_once is applied_twice

    @pytest.mark.asyncio
    async def test_aiter(self) -> None:
        ag = _agen_copy(1, 2)
        res = [i async for i in aiter(ag)]
        assert res == [1, 2]

    @pytest.mark.asyncio
    async def test_aiter_class(self) -> None:
        results = []
        ag = decorate(_AsyncIterable)()
        ait = aiter(ag)
        while True:
            try:
                results.append(await anext(ait))
            except StopAsyncIteration:
                break

        assert results == [1, 2]


class TestAsyncGenAnext:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_1(self, anext_: Anext) -> None:
        ag = _agen_copy(1, 2)
        assert await anext_(ag) == 1
        assert await anext_(ag) == 2
        assert await anext_(ag, 'buckle my shoe') == 'buckle my shoe'
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_2(self, anext_: Anext) -> None:
        ag = _agen_copy(1, 2)
        assert await anext_(ag) == 1
        assert await anext_(ag) == 2
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_3(self, anext_: Anext) -> None:
        ag = _agen_copy(1, 2)
        assert await anext_(ag, 'default') == 1
        assert await anext_(ag, 'default') == 2
        assert await anext_(ag, 'default') == 'default'
        assert await anext_(ag, 'default') == 'default'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_4_send(self, anext_: Anext) -> None:
        ag = _agen_copy(1, 2)
        cr = anext_(ag, 'completed')
        with (
            pytest.raises(StopIteration),
            contextlib.closing(cr.__await__()) as g,
        ):
            g.send(None)

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_bad_throw(self, anext_: Anext) -> None:
        ag = _agen_copy(1, 2)
        cr = anext_(ag, 'completed')
        with pytest.raises(TypeError):
            cr.throw()  # type: ignore[call-overload]
        cr.close()


class TestAsyncGenAnextIter:
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_1(self, anext_: Anext) -> None:
        ag = _agen_aw_exc_aw(MyError, x1=1, x2=2)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            assert g.throw(MyError()) == 2
            with pytest.raises(
                StopIteration, check=lambda e: e.value == 'default'
            ):
                g.send(None)

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_2(self, anext_: Anext) -> None:
        ag = _agen_aw_exc_aw(MyError, x1=1, x2=2)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            assert g.throw(MyError()) == 2
            with pytest.raises(MyError):
                g.throw(MyError())

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_3(self, anext_: Anext) -> None:
        ag = _agen_aw_exc_aw(MyError, x1=1, x2=2)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            g.close()
            with pytest.raises(RuntimeError, match='cannot reuse'):
                assert g.send(None) == 1

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_4(self, anext_: Anext) -> None:
        @decorate
        async def agenfn_aw_exc_aw() -> AsyncGenerator[Never, Any]:
            try:
                await _old_suspend_suspend_return(1)
            except MyError:
                await _old_suspend_suspend_return(2)
            return
            yield

        ag = agenfn_aw_exc_aw()
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 10
            assert g.throw(MyError()) == 20
            with pytest.raises(MyError, match='val'):
                g.throw(MyError('val'))

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_5(self, anext_: Anext) -> None:
        @decorate
        async def agenfn_aw_ex_yi() -> AsyncGenerator[Literal['aaa'], Any]:
            try:
                await _old_suspend_suspend_return(1)
            except MyError:
                return
            yield 'aaa'

        ag = agenfn_aw_ex_yi()
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 10
            with pytest.raises(StopIteration, match='default'):
                g.throw(MyError())

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_6(self, anext_: Anext) -> None:
        @decorate
        async def agenfn() -> AsyncGenerator[Literal['aaa'], Any]:
            await _old_suspend_suspend_return(1)
            yield 'aaa'

        ag = agenfn()
        with (
            contextlib.closing(anext_(ag, 'default').__await__()) as g,
            pytest.raises(MyError),
        ):
            g.throw(MyError())
        g.close()


class TestAsyncGenAsyncio:
    @pytest.mark.asyncio
    async def test_01(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 2], Any]:
            yield 1
            await asyncio.sleep(0.01)
            yield 2
            await asyncio.sleep(0.01)
            return

        res = [x async for x in agen()]
        assert res == [1, 2]

    @pytest.mark.asyncio
    async def test_02(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 2], Any]:
            yield 1
            await asyncio.sleep(0.01)
            yield 2
            raise ZeroDivisionError

        with pytest.raises(ZeroDivisionError):
            async for _ in agen():
                pass

    @pytest.mark.asyncio
    async def test_03(self) -> None:
        class Gen:
            async def __aiter__(self) -> AsyncGenerator[Literal[1, 2], Any]:
                yield 1
                await asyncio.sleep(0.01)
                yield 2

        res = [x async for x in decorate(Gen)()]
        assert res == [1, 2]

    @pytest.mark.asyncio
    async def test_anext_01(self) -> None:
        ait = _async_gen_asyncio_anext().__aiter__()

        assert await ait.__anext__() == 1
        assert await ait.__anext__() == 2
        assert await ait.__anext__() == 3
        assert await ait.__anext__() == 4
        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()
        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()

    @pytest.mark.asyncio
    async def test_anext_02(self) -> None:
        ait = _async_gen_asyncio_anext().__aiter__()

        assert await ait.__anext__() == 1
        assert await ait.__anext__() == 2
        with pytest.raises(StopIteration, check=lambda e: e.value == 1000):
            ait.__anext__().__await__().throw(ZeroDivisionError)
        assert await ait.__anext__() == 4
        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()

    @pytest.mark.asyncio
    async def test_anext_03(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[Any | Literal[1], Any]:
            v = yield 1
            v = yield v
            yield v * 100

        ait = foo().__aiter__()

        with pytest.raises(StopIteration, check=lambda e: e.value == 1):
            ait.__anext__().__await__().send(None)

        with pytest.raises(StopIteration, check=lambda e: e.value == 10):
            ait.__anext__().__await__().send(10)

        with pytest.raises(StopIteration, check=lambda e: e.value == 1200):
            ait.__anext__().__await__().send(12)

        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()

    @pytest.mark.asyncio
    async def test_anext_04(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[None, Any]:
            nonlocal done
            try:
                yield
            except:  # noqa: E722
                pass
            done = 1

        ag = agen()
        await ag.asend(None)
        with pytest.raises(StopAsyncIteration):
            await ag.asend(None)
        assert done == 1

    @pytest.mark.asyncio
    async def test_anext_tuple(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[tuple[Literal[1, 2]], Any]:
            try:
                yield (1,)
            except ZeroDivisionError:
                yield (2,)

        ait = foo().__aiter__()

        assert await ait.__anext__() == (1,)
        with pytest.raises(StopIteration, check=lambda e: e.value == (2,)):
            ait.__anext__().__await__().throw(ZeroDivisionError)
        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()

    @pytest.mark.asyncio
    async def test_anext_tuple_no_exceptions(self) -> None:
        # StopAsyncIteration exceptions should be cleared.
        # See: https://github.com/python/cpython/issues/128078.
        @decorate
        async def foo() -> AsyncGenerator[Never, Any]:
            if False:
                yield (1, 2)

        ait = foo().__aiter__()
        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()
        res = await anext(ait, ('a', 'b'))
        assert res == ('a', 'b')

    @pytest.mark.asyncio
    async def test_anext_stopiteration(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[StopIteration, Any]:
            try:
                yield StopIteration(1)
            except ZeroDivisionError:
                yield StopIteration(3)

        ait = foo().__aiter__()

        v = await ait.__anext__()
        assert isinstance(v, StopIteration)
        assert v.value == 1

        with pytest.raises(
            StopIteration,
            check=lambda e: (
                isinstance(e.value, StopIteration) and e.value.value == 3
            ),
        ):
            ait.__anext__().__await__().throw(ZeroDivisionError)

        with pytest.raises(StopAsyncIteration):
            await ait.__anext__()

    @pytest.mark.asyncio
    async def test_aclose_01(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 12], Any]:
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                await asyncio.sleep(0.01)
                yield 12

        ag = agen()
        ait = ag.__aiter__()
        await ait.__anext__()

        with pytest.raises(
            RuntimeError, match='async generator ignored GeneratorExit'
        ):
            await ag.aclose()

    @pytest.mark.asyncio
    async def test_aclose_02(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[Literal[1], Any]:
            nonlocal done
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done += 1
            done += 1000

        ag = agen()
        ait = ag.__aiter__()
        await ait.__anext__()
        await ag.aclose()
        assert done == 1

    @pytest.mark.asyncio
    async def test_aclose_03(self) -> None:
        done = 0
        fut = asyncio.Future[None]()

        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 2], Any]:
            nonlocal done
            try:
                yield 1
                await fut
                done += 1000
                yield 2
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done += 1
            done += 1000

        ag = agen()
        ait = ag.__aiter__()
        assert await ait.__anext__() == 1

        await ag.aclose()
        assert done == 1

        # Silence ResourceWarnings
        fut.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_gc_aclose_04(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[Literal[1], Any]:
            nonlocal done
            try:
                while True:
                    yield 1
            finally:
                await asyncio.sleep(0)
                done = 1

        ag = agen()
        await ag.__anext__()
        await ag.__anext__()
        del ag
        gc_collect()  # For PyPy or other GCs.

        # Starts running the aclose task
        await asyncio.sleep(0.1)
        assert done == 1

    @pytest.mark.asyncio
    async def test_aclose_05(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[None, Any]:
            nonlocal done
            try:
                yield
            except:  # noqa: E722
                pass
            done = 1

        ag = agen()
        await ag.asend(None)
        await ag.aclose()
        assert done == 1

    @pytest.mark.asyncio
    async def test_aclose_06(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[None, Any]:
            nonlocal done
            try:
                yield
            except:  # noqa: E722
                pass
            yield
            done += 1

        ag = agen()
        await ag.asend(None)
        with pytest.raises(RuntimeError, match='ignored GeneratorExit'):
            await ag.aclose()
        assert done == 0

    @pytest.mark.asyncio
    async def test_aclose_07(self) -> None:
        done = 0

        @decorate
        async def target() -> None:
            await asyncio.sleep(0.01)
            raise ZeroDivisionError

        @decorate
        async def agen() -> AsyncGenerator[Literal[1], Any]:
            nonlocal done
            task = asyncio.create_task(target())
            try:
                yield 1
            finally:
                try:
                    await task
                except ZeroDivisionError:
                    done = 1

        ag = agen()
        ait = ag.__aiter__()
        await ait.__anext__()
        await ag.aclose()
        assert done == 1

    @pytest.mark.asyncio
    async def test_asend_01(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[Any | Literal[1], Any]:
            nonlocal done
            try:
                await asyncio.sleep(0.01)
                v = yield 1
                await asyncio.sleep(0.01)
                yield v * 2
                await asyncio.sleep(0.01)
                return
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done = 1

        ag = agen()
        assert await ag.asend(None) == 1
        assert await ag.asend(100) == 200
        with pytest.raises(StopAsyncIteration):
            await ag.asend(None)
        assert done == 1

    @pytest.mark.asyncio
    async def test_asend_02(self) -> None:
        done = 0

        @decorate
        async def sleep_n_crash(delay) -> Never:
            await asyncio.sleep(delay)
            raise ZeroDivisionError

        @decorate
        async def agen() -> AsyncGenerator[Literal[1], Any]:
            nonlocal done
            try:
                await asyncio.sleep(0.01)
                v = yield 1
                await sleep_n_crash(0.01)
                done += 1000
                yield v * 2
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done = 1

        ag = agen()
        assert await ag.asend(None) == 1
        with pytest.raises(ZeroDivisionError):
            await ag.asend(100)
        assert done == 1

    @pytest.mark.asyncio
    async def test_asend_03(self) -> None:
        done = 0
        loop = asyncio.get_running_loop()

        @decorate
        async def sleep_n_crash(delay) -> None:
            fut = asyncio.ensure_future(asyncio.sleep(delay), loop=loop)
            loop.call_later(delay / 2, fut.cancel)
            return await fut

        @decorate
        async def agen() -> AsyncGenerator[Any | Literal[1], Any]:
            nonlocal done
            try:
                await asyncio.sleep(0.01)
                v = yield 1
                await sleep_n_crash(0.01)
                done += 1000
                yield v * 2
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done = 1

        ag = agen()
        assert await ag.asend(None) == 1
        with pytest.raises(asyncio.CancelledError):
            await ag.asend(100)
        assert done == 1

    @pytest.mark.asyncio
    async def test_athrow_01(self) -> None:
        done = 0

        @decorate
        async def agen() -> AsyncGenerator[Any | int, Any]:
            nonlocal done
            try:
                await asyncio.sleep(0.01)
                try:
                    v = yield 1
                except MyError:
                    v = 1000
                    await asyncio.sleep(0.01)
                yield v * 2
                await asyncio.sleep(0.01)
                # return
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done = 1

        ag = agen()
        assert await ag.asend(None) == 1
        assert await ag.athrow(MyError) == 2000
        with pytest.raises(StopAsyncIteration):
            await ag.asend(None)
        assert done == 1

    @pytest.mark.asyncio
    async def test_athrow_02(self) -> None:
        done = 0

        @decorate
        async def sleep_n_crash(delay) -> None:
            fut = asyncio.ensure_future(asyncio.sleep(delay))
            asyncio.get_running_loop().call_later(delay / 2, fut.cancel)
            return await fut

        @decorate
        async def agen() -> AsyncGenerator[Any | Literal[1], Any]:
            nonlocal done
            try:
                await asyncio.sleep(0.01)
                try:
                    v = yield 1
                except MyError:
                    await sleep_n_crash(0.01)
                    raise AssertionError from None

                yield v * 2
                await asyncio.sleep(0.01)
                # return
            finally:
                await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
                done = 1

        ag = agen()
        assert await ag.asend(None) == 1
        with pytest.raises(asyncio.CancelledError):
            await ag.athrow(MyError)
        assert done == 1

    @pytest.mark.asyncio
    async def test_athrow_tuple(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[
            tuple[Literal[2]] | Literal[1], Any
        ]:
            try:
                yield 1
            except ZeroDivisionError:
                yield (2,)

        ag = agen()
        assert await ag.asend(None) == 1
        assert await ag.athrow(ZeroDivisionError) == (2,)
        with pytest.raises(StopAsyncIteration):
            await ag.asend(None)

    @pytest.mark.asyncio
    async def test_athrow_stopiteration(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[StopIteration | Literal[1], Any]:
            try:
                yield 1
            except ZeroDivisionError:
                yield StopIteration(2)

        ag = agen()
        assert await ag.asend(None) == 1

        v = await ag.athrow(ZeroDivisionError)
        assert isinstance(v, StopIteration)
        assert v.value == 2

        with pytest.raises(StopAsyncIteration):
            await ag.asend(None)

    @pytest.mark.asyncio
    async def test_shutdown_01(self) -> None:
        finalized = 0

        @decorate
        async def waiter(timeout: float) -> AsyncGenerator[Literal[1], Any]:
            nonlocal finalized
            try:
                await asyncio.sleep(timeout)
                yield 1
            finally:
                await asyncio.sleep(0)
                finalized += 1

        @decorate
        async def wait() -> None:
            async for _ in waiter(1):
                pass

        t1 = asyncio.create_task(wait())
        t2 = asyncio.create_task(wait())

        await asyncio.sleep(0.1)

        # Silence warnings
        t1.cancel()
        t2.cancel()

        with pytest.raises(asyncio.CancelledError):
            await t1
        with pytest.raises(asyncio.CancelledError):
            await t2

        await asyncio.get_running_loop().shutdown_asyncgens()
        assert finalized == 2

    @pytest.mark.asyncio
    async def test_shutdown_02(self) -> None:
        # See https://bugs.python.org/issue38013
        messages = []
        ag = _agen_copy(1, 2)
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: messages.append(context))

        async for _ in ag:
            break
        assert messages == []
        gc_collect()


class TestAsyncGenExpression:
    @pytest.mark.asyncio
    async def test_1(self) -> None:
        @decorate
        async def arange(n) -> AsyncGenerator[int, Any]:
            for i in range(n):
                await asyncio.sleep(0.01)
                yield i

        @decorate
        def make_arange(n) -> AsyncGenerator[int, Any]:
            # This syntax is legal starting with Python 3.7
            return (i * 2 async for i in arange(n))

        res = [i async for i in make_arange(10)]
        assert res == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_2(self) -> None:
        @decorate
        async def wrap(n: int) -> int:
            await asyncio.sleep(0.01)
            return n

        def make_arange(n: int) -> AsyncGenerator[int, Any]:
            # This syntax is legal starting with Python 3.7
            return (i * 2 for i in range(n) if await wrap(i))

        res = [i async for i in make_arange(10)]
        assert res == [i * 2 for i in range(1, 10)]


class TestAsyncReuseAlreadyAwaited:
    @pytest.mark.asyncio
    async def test_await_same_anext(self) -> None:
        ag = _agen_copy(1, 2)
        cr = ag.__anext__()
        await cr
        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            await cr

        await ag.aclose()  # prevent unfinished iterator warning

    @pytest.mark.asyncio
    async def test_await_same_aclose(self) -> None:
        ag = _agen_copy(1, 2)
        cr = ag.aclose()
        await cr
        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            await cr

    def test_throw_same_aclose(self) -> None:
        ag = _agen_copy(1, 2)
        cr = ag.aclose()
        with pytest.raises(StopIteration):
            cr.throw(GeneratorExit)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            cr.throw(GeneratorExit)

    def test_throw_custom_same_aclose(self) -> None:
        ag = _agen_copy(1, 2)
        cr = ag.aclose()
        with pytest.raises(MyError):
            cr.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            cr.throw(MyError)

    def test_throw_custom_same_athrow(self) -> None:
        ag = _agen_copy(1, 2)
        cr = ag.athrow(MyError)
        with pytest.raises(MyError):
            cr.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            cr.throw(MyError)

    @pytest.mark.asyncio
    async def test_aclose_with_different_coros(self) -> None:
        # Regression test for https://bugs.python.org/issue39606
        ag = _agen_copy(1, 2)
        await ag.aclose()
        await ag.aclose()


class TestAsyncGenAclose:
    @pytest.mark.asyncio
    async def test_after_exhaustion(self) -> None:
        # Regression test for https://bugs.python.org/issue39606
        ag = _agen_copy(1, 2)
        async for _ in ag:
            pass
        await ag.aclose()

    @pytest.mark.asyncio
    async def test_compatible_with_get_stack(self) -> None:
        ag = _agen_copy(object())
        asyncio.create_task(ag.aclose())  # noqa: RUF006
        for task in asyncio.all_tasks():
            # No AttributeError raised
            task.get_stack()

    def test_throw(self) -> None:
        ag = _agen_copy()
        with pytest.raises(MyError):
            ag.aclose().throw(MyError)

        del ag
        gc_collect()  # does not warn unawaited


class TestAsyncGenNeverAwaited:
    def test_asend(self) -> None:
        # gh-113753: asend objects allocated from a free-list should warn.
        # Ensure there is a finalized 'asend' object ready to be reused.
        try:
            ag = _agen_copy(1, 2)
            ag.asend(None).send(None)
        except StopIteration:
            pass

        ag = _agen_copy()
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'asend' of '.*' was never awaited",
        ):
            ag.asend(None)  # type: ignore[unused-coroutine]

    def test_athrow(self) -> None:
        ag = _agen_copy(1, 2)
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'athrow' of '.*' was never awaited",
        ):
            ag.athrow(RuntimeError)  # type: ignore[unused-coroutine]

    def test_aclose(self) -> None:
        ag = _agen_copy(1, 2)
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'aclose' of '.*' was never awaited",
        ):
            ag.aclose()  # type: ignore[unused-coroutine]


class TestAsyncGenAlreadyRunning:
    def test_asend_send(self) -> None:
        ag = old_agenfn_inf_aw()
        cr = ag.asend(None)
        cr.send(None)
        cr2 = ag.asend(None)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            cr2.send(None)

        del cr2
        gc_collect()  # does not warn unawaited

    def test_athrow_send(self) -> None:
        ag = old_agenfn_inf_aw()
        cr = ag.asend(None)
        cr.send(None)
        cr2 = ag.athrow(Exception)

        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            cr2.send(None)

        del cr2
        gc_collect()  # does not warn unawaited
