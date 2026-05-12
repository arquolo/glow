import asyncio
import contextlib
import gc
import inspect
import sys
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
from typing import (
    Any,
    Literal,
    Never,
    NoReturn,
    Protocol,
    Self,
    cast,
    overload,
)

import pytest

from glow import time_this

decorate = time_this()
# decorate = time_this(disable=True)


class AwaitError(Exception):
    pass


class MyError(Exception):
    pass


class BadBaseError(Exception):
    def __new__(cls, *args, **kwargs) -> type[Self]:  # type: ignore[misc]
        return cls


def gc_collect() -> None:
    gc.collect()
    gc.collect()
    gc.collect()


# --------------------------------- mixtures ---------------------------------


def as_agen[Y, S](obj: AsyncGenerator[Y, S]) -> AsyncGeneratorType[Y, S]:
    return cast('AsyncGeneratorType[Y, S]', obj)


def as_gen[Y, S, R](obj: Generator[Y, S, R]) -> GeneratorType[Y, S, R]:
    return cast('GeneratorType[Y, S, R]', obj)


def as_coro[Y, S, R](obj: Coroutine[Y, S, R]) -> CoroutineType[Y, S, R]:
    return cast('CoroutineType[Y, S, R]', obj)


def throw(exc: type[Exception]) -> NoReturn:
    raise exc


@decorate
def gen_none() -> Generator[None, Any]:
    yield


@decorate
def gen_1() -> Generator[Literal[1], Any]:
    yield 1


@decorate
def gen_return_42() -> Generator[Never, Any, Literal[42]]:
    return 42
    yield


@decorate
def gen_yield_raises[T](
    value: T, exc_tp: type[Exception] = ValueError
) -> Generator[T, Any, NoReturn]:
    yield value
    raise exc_tp


@decorate
def gen_raises(exc_tp: type[Exception]) -> Generator[Never, Any, NoReturn]:
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
async def agen_none() -> AsyncGenerator[None, Any]:
    yield


@decorate
async def coro_none() -> None:
    pass


@coroutine
def old_suspend[S: str](v: S = 'result') -> Generator[tuple[S]]:
    yield (v,)


@coroutine
def _old_suspend_return[Y, S](v: Y) -> Generator[Y, S, S]:
    send = yield v
    return send


@coroutine
def _old_suspend_suspend_return[S](v: int) -> Generator[int, S, S]:
    yield v * 10
    send = yield v * 10 + 1
    return send


@decorate
async def old_agenfn_inf_aw_exc() -> AsyncGenerator[Never, Any]:
    while True:
        try:
            await _old_suspend_return(None)
        except MyError:
            pass
    return
    yield


@decorate
async def old_agenfn_yi_inf_aw_exc() -> AsyncGenerator[None, Any]:
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
async def old_agenfn_aw_exc_aw(
    exc_type, x1=None, x2=None
) -> AsyncGenerator[Never, Any]:
    try:
        await _old_suspend_return(x1)
    except exc_type:
        await _old_suspend_return(x2)
    return
    yield


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
def agen_copy() -> AsyncGenerator[Never, Any]: ...
@overload
def agen_copy[T](*values: T) -> AsyncGenerator[T, Any]: ...


@decorate
async def agen_copy[T](*values: T) -> AsyncGenerator[T, Any]:
    for x in values:
        yield x


@decorate
def gen_copy[T](*values: T) -> Generator[T]:
    for x in values:  # noqa: UP028
        yield x


@decorate
async def agen_yi_raise(
    tp: type[BaseException],
) -> AsyncGenerator[Literal[123], Any]:
    yield 123
    raise tp


@decorate
async def _async_gen_asyncio_anext() -> (
    AsyncGenerator[Literal[1, 2, 3, 4, 1000], Any]
):
    yield 1
    await asyncio.sleep(0.01)
    try:
        yield 2
        yield 3
    except ZeroDivisionError:
        yield 1000
    await asyncio.sleep(0.01)
    yield 4


def to_list[T](ai: AsyncIterator[T]) -> list[T]:
    @decorate
    async def iterate() -> list[T]:
        return [x async for x in ai]

    coro = iterate()
    exc = False
    while True:
        try:
            fut = coro.throw(AwaitError) if exc else coro.send(None)
        except StopIteration as ex:
            return ex.value
        exc = fut == ('throw',)


class _Empty(Enum):
    token = 0


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


def _sync_iterate[T](g: Iterator[T]) -> Generator[T | str]:
    while True:
        try:
            yield g.__next__()
        except StopIteration:
            yield 'STOP'
            return
        except Exception as ex:  # noqa: BLE001
            yield str(type(ex))


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
                yield ex.args[0] if ex.args else 'EMPTY StopIteration'
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


class TestGen:
    def test_send_non_none_to_new_gen(self) -> None:
        g = gen_1()
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

        @decorate
        def cb(*args) -> None:
            inspect.stack()

        thresholds = gc.get_threshold()

        gc.callbacks.append(cb)
        gc.set_threshold(1, 0, 0)
        try:
            gen_1()
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
            gen_1()
        finally:
            gc.set_threshold(*thresholds)

    def test_ag_frame_f_back(self) -> None:
        ag = as_agen(agen_none())
        assert ag.ag_frame
        assert ag.ag_frame.f_back is None

    @pytest.mark.asyncio
    async def test_cr_frame_f_back(self) -> None:
        cr = as_coro(coro_none())
        assert cr.cr_frame
        assert cr.cr_frame.f_back is None
        await cr

    def test_gi_frame_f_back(self) -> None:
        gi = as_gen(gen_none())
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
        with pytest.raises(ValueError) as cm:
            next(make)
        assert cm.value.__context__ is None

        assert sys.exception() is None

    @decorate
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
            try:
                assert sys.exception() is None
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
        g = gen_none()
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
        with pytest.raises(StopIteration) as cm:
            g.send((2,))
        assert cm.value.value == (2,)

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
        g = gen_none()
        g.send(None)
        assert g.close() is None

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
        def gen() -> Generator[None, Any, NoReturn]:
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
                v = 1000
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
                yield from gen_none()

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
                    yield from gen_none()
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
        def gen() -> Generator[NoReturn, Any, NoReturn]:
            yield throw(StopIteration)

        with pytest.raises(
            RuntimeError, match='generator raised StopIteration'
        ):
            next(gen())

    def test_stopiteration_wrapping_context(self) -> None:
        @decorate
        def gen() -> Generator[NoReturn, Any, NoReturn]:
            yield throw(StopIteration)

        with pytest.raises(
            RuntimeError,
            check=lambda exc: (
                type(exc.__cause__) is StopIteration
                and type(exc.__context__) is StopIteration
                and exc.__suppress_context__
            ),
        ):
            next(gen())


class TestAsyncGenIteration:
    def test_1(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123, 456, 789], Any]:
            await old_suspend()
            a = yield 123
            assert a is None
            await old_suspend()
            yield 456
            await old_suspend()
            yield 789

        assert to_list(agen()) == [123, 456, 789]

    def test_2(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123], Any]:
            await old_suspend()
            yield 123
            await old_suspend()

        ag = agen()
        ai = ag.__aiter__()

        an = ai.__anext__()
        assert an.__await__().__next__() == ('result',)

        with pytest.raises(StopIteration, check=lambda e: e.value == 123):
            an.__await__().__next__()

        an = ai.__anext__()
        assert an.__await__().__next__() == ('result',)

        with pytest.raises(StopAsyncIteration, check=lambda e: not e.args):
            an.__await__().__next__()


class TestAsyncGenException:
    def test_1(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123, 456], Any]:
            await old_suspend()
            yield 123
            await old_suspend('throw')
            yield 456

        with pytest.raises(AwaitError):
            to_list(agen())

    def test_2(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[123], Any]:
            await old_suspend()
            yield 123
            raise ZeroDivisionError

        ag = agen()
        ai = ag.__aiter__()
        an = ai.__anext__()
        assert an.__await__().__next__() == ('result',)

        with pytest.raises(StopIteration, check=lambda e: e.value == 123):
            an.__await__().__next__()

        with pytest.raises(ZeroDivisionError):
            ai.__anext__().__await__().__next__()

    def test_3(self) -> None:
        with pytest.raises(
            RuntimeError, match=r'async generator.*StopAsyncIteration'
        ):
            to_list(agen_yi_raise(StopAsyncIteration))

    def test_4(self) -> None:
        with pytest.raises(
            RuntimeError, match=r'async generator.*StopIteration'
        ):
            to_list(agen_yi_raise(StopIteration))

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
                await old_suspend()
            finally:
                await old_suspend()
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
                await old_suspend()
                yield 1
                raise ZeroDivisionError
            finally:
                yield 2
                await old_suspend()
                yield 3

            yield 100

        gencmp(sync_gen(), async_gen())

    def test_8(self) -> None:
        with pytest.raises(
            TypeError, match=r'non-None value .* async generator'
        ):
            agen_copy().__anext__().send(100)

    def test_9(self) -> None:
        @decorate
        def sync_gen_wrapper() -> Iterator[Literal[10, 20, 30]]:
            yield 10
            g = gen_copy(1, 2)
            g.send(None)
            try:
                g.throw(GeneratorExit())
            except GeneratorExit:
                yield 20
            yield 30

        @decorate
        async def async_gen_wrapper() -> AsyncIterator[Literal[10, 20, 30]]:
            yield 10
            asg = agen_copy(1, 2)
            await asg.asend(None)
            try:
                await asg.athrow(GeneratorExit())
            except GeneratorExit:
                yield 20
            yield 30

        gencmp(sync_gen_wrapper(), async_gen_wrapper())

    def test_10(self) -> None:
        @decorate
        async def agen() -> AsyncIterator[Literal[123]]:
            with pytest.raises(
                RuntimeError,
                match=r'anext\(\): asynchronous generator is already running',
            ):
                await anext(me)
            yield 123

        me = agen()
        ai = me.__aiter__()
        an = ai.__anext__()

        with pytest.raises(StopIteration):
            an.__await__().__next__()

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            an.__await__().send(None)

    @pytest.mark.asyncio
    @decorate
    async def test_12_async(self) -> None:
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


class TestOldAsyncGen:
    def test_asend_throw_concurrent_with_send(self) -> None:
        ag = old_agenfn_inf_aw_exc()
        g = ag.asend(None)
        g.send(None)
        g2 = ag.asend(None)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            g2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            g2.send(None)

    def test_athrow_throw_concurrent_with_send(self) -> None:
        ag = old_agenfn_inf_aw_exc()
        g = ag.asend(None)
        g.send(None)
        g2 = ag.athrow(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            g2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            g2.send(None)

    def test_asend_throw_concurrent_with_throw(self) -> None:
        ag = old_agenfn_yi_inf_aw_exc()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)

        g = ag.athrow(MyError)
        g.throw(MyError)
        g2 = ag.asend(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            g2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            g2.send(None)

    def test_athrow_throw_concurrent_with_throw(self) -> None:
        ag = old_agenfn_yi_inf_aw_exc()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)

        g = ag.athrow(MyError)
        g.throw(MyError)

        g2 = ag.athrow(MyError)
        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            g2.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            g2.send(None)

    def test_asend_close_runtime_error(self) -> None:
        ag = old_agenfn_aw_exc_aw(GeneratorExit)
        g = ag.asend(None)
        g.send(None)
        with pytest.raises(
            RuntimeError, match='coroutine ignored GeneratorExit'
        ):
            g.close()

    def test_athrow_close_runtime_error(self) -> None:
        ag = old_agenfn_yi_aw_aw()
        with pytest.raises(StopIteration):
            ag.asend(None).send(None)
        g = ag.athrow(MyError)
        g.send(None)
        with pytest.raises(
            RuntimeError, match='coroutine ignored GeneratorExit'
        ):
            g.close()


class TestAsyncGenApi:
    def test_1(self) -> None:
        ag = cast('AsyncGeneratorType[int, Any]', agen_copy(1, 2))

        assert ag.ag_await is None
        assert isinstance(ag.ag_frame, FrameType)
        assert not ag.ag_running
        assert isinstance(ag.ag_code, CodeType)

        aclose = ag.aclose()
        assert inspect.isawaitable(aclose)
        aclose.close()

    def test_aiter_idempotent(self) -> None:
        applied_once = aiter(agen_copy())
        applied_twice = aiter(applied_once)
        assert applied_once is applied_twice

    @pytest.mark.asyncio
    @decorate
    async def test_aiter(self) -> None:
        ag = agen_copy(1, 2)
        res = [i async for i in aiter(ag)]
        assert res == [1, 2]

    @pytest.mark.asyncio
    @decorate
    async def test_aiter_class(self) -> None:
        results = []

        class Gen:
            async def __aiter__(self) -> AsyncIterator[Literal[1, 2]]:
                yield 1
                yield 2

        g = decorate(Gen)()

        ait = aiter(g)
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
        ag = agen_copy(1, 2)
        assert await anext_(ag) == 1
        assert await anext_(ag) == 2
        assert await anext_(ag, 'buckle my shoe') == 'buckle my shoe'
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_2(self, anext_: Anext) -> None:
        ag = agen_copy(1, 2)
        assert await anext_(ag) == 1
        assert await anext_(ag) == 2
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)
        with pytest.raises(StopAsyncIteration):
            await anext_(ag)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_3(self, anext_: Anext) -> None:
        ag = agen_copy(1, 2)
        assert await anext_(ag, 'default') == 1
        assert await anext_(ag, 'default') == 2
        assert await anext_(ag, 'default') == 'default'
        assert await anext_(ag, 'default') == 'default'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    async def test_4_send(self, anext_: Anext) -> None:
        ag = agen_copy(1, 2)
        an = anext_(ag, 'completed')
        with (
            pytest.raises(StopIteration),
            contextlib.closing(an.__await__()) as ag2,
        ):
            ag2.send(None)

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_bad_throw(self, anext_: Anext) -> None:
        ag = agen_copy(1, 2)
        an = anext_(ag, 'completed')
        with pytest.raises(TypeError):
            an.throw()  # type: ignore[call-overload]
        an.close()


class TestAnextIter:
    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_1(self, anext_: Anext) -> None:
        ag = old_agenfn_aw_exc_aw(MyError, x1=1, x2=2)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            assert g.throw(MyError()) == 2
            with pytest.raises(
                StopIteration, check=lambda e: e.value == 'default'
            ):
                g.send(None)

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_2(self, anext_: Anext) -> None:
        ag = old_agenfn_aw_exc_aw(MyError, x1=1, x2=2)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            assert g.throw(MyError()) == 2
            with pytest.raises(MyError):
                g.throw(MyError())

    @pytest.mark.parametrize('anext_', [py_anext, anext])
    def test_3(self, anext_: Anext) -> None:
        ag = old_agenfn_aw_exc_aw(MyError, x1=1, x2=2)
        print(ag)
        with contextlib.closing(anext_(ag, 'default').__await__()) as g:
            assert g.send(None) == 1
            g.close()
            print(g)
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
        an = anext_(ag, 'default')
        with (
            contextlib.closing(an.__await__()) as g,
            pytest.raises(MyError),
        ):
            g.throw(MyError())
        an.close()


class TestAsyncGenAsyncio:
    @pytest.mark.asyncio
    @decorate
    async def test_01(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 2], Any]:
            yield 1
            await asyncio.sleep(0.01)
            yield 2
            await asyncio.sleep(0.01)
            return
            yield 3

        res = [x async for x in agen()]
        assert res == [1, 2]

    @pytest.mark.asyncio
    @decorate
    async def test_02(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 2], Any]:
            yield 1
            await asyncio.sleep(0.01)
            yield 2
            raise ZeroDivisionError
            yield 3

        with pytest.raises(ZeroDivisionError):
            async for _ in agen():
                pass

    @pytest.mark.asyncio
    @decorate
    async def test_03(self) -> None:
        class Gen:
            async def __aiter__(self) -> AsyncGenerator[Literal[1, 2], Any]:
                yield 1
                await asyncio.sleep(0.01)
                yield 2

        res = [x async for x in decorate(Gen)()]
        assert res == [1, 2]

    @pytest.mark.asyncio
    @decorate
    async def test_anext_04_1(self) -> None:
        it = _async_gen_asyncio_anext().__aiter__()

        assert await it.__anext__() == 1
        assert await it.__anext__() == 2
        assert await it.__anext__() == 3
        assert await it.__anext__() == 4
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    @decorate
    async def test_anext_04_2(self) -> None:
        it = _async_gen_asyncio_anext().__aiter__()

        assert await it.__anext__() == 1
        assert await it.__anext__() == 2
        with pytest.raises(StopIteration, check=lambda e: e.args[0] == 1000):
            it.__anext__().__await__().throw(ZeroDivisionError)
        assert await it.__anext__() == 4
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    @decorate
    async def test_anext_05(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[Any | Literal[1], Any]:
            v = yield 1
            v = yield v
            yield v * 100

        it = foo().__aiter__()

        with pytest.raises(StopIteration, check=lambda e: e.value == 1):
            it.__anext__().__await__().send(None)

        with pytest.raises(StopIteration, check=lambda e: e.value == 10):
            it.__anext__().__await__().send(10)

        with pytest.raises(StopIteration, check=lambda e: e.value == 1200):
            it.__anext__().__await__().send(12)

        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    @decorate
    async def test_anext_06(self) -> None:
        done = 0

        # test synchronous generators
        def foo() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass

        ag = foo()
        ag.send(None)
        with pytest.raises(StopIteration):
            ag.send(None)

        # now with asynchronous generators

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
    @decorate
    async def test_anext_tuple(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[tuple[Literal[1, 2]], Any]:
            try:
                yield (1,)
            except ZeroDivisionError:
                yield (2,)

        it = foo().__aiter__()

        assert await it.__anext__() == (1,)
        with pytest.raises(StopIteration) as cm:
            it.__anext__().__await__().throw(ZeroDivisionError)
        assert cm.value.args[0] == (2,)
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    @decorate
    async def test_anext_tuple_no_exceptions(self) -> None:
        # StopAsyncIteration exceptions should be cleared.
        # See: https://github.com/python/cpython/issues/128078.
        @decorate
        async def foo() -> AsyncGenerator[Never, Any]:
            if False:
                yield (1, 2)

        it = foo().__aiter__()
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
        res = await anext(it, ('a', 'b'))
        assert res == ('a', 'b')

    @pytest.mark.asyncio
    @decorate
    async def test_anext_stopiteration(self) -> None:
        @decorate
        async def foo() -> AsyncGenerator[StopIteration, Any]:
            try:
                yield StopIteration(1)
            except ZeroDivisionError:
                yield StopIteration(3)

        it = foo().__aiter__()

        v = await it.__anext__()
        assert isinstance(v, StopIteration)
        assert v.value == 1

        with pytest.raises(
            StopIteration,
            check=lambda e: (
                isinstance(e.value, StopIteration) and e.value.value == 3
            ),
        ):
            it.__anext__().__await__().throw(ZeroDivisionError)

        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    @decorate
    async def test_aclose_06(self) -> None:
        @decorate
        async def agen() -> AsyncGenerator[Literal[1, 12], Any]:
            try:
                yield 1
                raise ZeroDivisionError
            finally:
                await asyncio.sleep(0.01)
                yield 12

        ag = agen()
        ai = ag.__aiter__()
        await ai.__anext__()

        with pytest.raises(
            RuntimeError, match='async generator ignored GeneratorExit'
        ):
            await ag.aclose()

    @pytest.mark.asyncio
    @decorate
    async def test_aclose_07(self) -> None:
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
        ai = ag.__aiter__()
        await ai.__anext__()
        await ag.aclose()
        assert done == 1

    @pytest.mark.asyncio
    @decorate
    async def test_aclose_08(self) -> None:
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
        ai = ag.__aiter__()
        assert await ai.__anext__() == 1

        await ag.aclose()
        assert done == 1

        # Silence ResourceWarnings
        fut.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    @decorate
    async def test_gc_aclose_09(self) -> None:
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
    @decorate
    async def test_aclose_10(self) -> None:
        done = 0

        # test synchronous generators
        def foo() -> Generator[None, Any]:
            try:
                yield
            except:  # noqa: E722
                pass

        ag = foo()
        ag.send(None)
        ag.close()

        # now with asynchronous generators

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
    @decorate
    async def test_aclose_11(self) -> None:
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
    @decorate
    async def test_aclose_12(self) -> None:
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
        ai = ag.__aiter__()
        await ai.__anext__()
        await ag.aclose()
        assert done == 1

    @pytest.mark.asyncio
    @decorate
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
    @decorate
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
    @decorate
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
    @decorate
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
    @decorate
    async def test_athrow_02(self) -> None:
        done = 0

        class FooError(Exception):
            pass

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
                except FooError:
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
            await ag.athrow(FooError)
        assert done == 1

    @pytest.mark.asyncio
    @decorate
    async def test_athrow_tuple(self) -> None:
        @decorate
        async def agen() -> (
            AsyncGenerator[tuple[Literal[2]] | Literal[1], Any]
        ):
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
    @decorate
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
    @decorate
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
    @decorate
    async def test_shutdown_02(self) -> None:
        messages = []
        ag = agen_copy(1, 2)
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: messages.append(context))

        async for _ in ag:
            break
        assert messages == []


class TestAsyncGenExpression:
    @pytest.mark.asyncio
    @decorate
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
    @decorate
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


@pytest.mark.asyncio
@decorate
async def test_async_gen_nonstarted_hooks_are_cancellable() -> None:
    # See https://bugs.python.org/issue38013
    messages = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _, context: messages.append(context)
    )
    async for _ in agen_copy(1, 2):
        break

    assert messages == []
    gc_collect()


class TestAsyncGenCalledTwice:
    @pytest.mark.asyncio
    @decorate
    async def test_await_same_anext(self) -> None:
        it = agen_copy(1, 2)
        nxt = it.__anext__()
        await nxt
        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited __anext__\(\)/asend\(\)',
        ):
            await nxt

        await it.aclose()  # prevent unfinished iterator warning

    @pytest.mark.asyncio
    @decorate
    async def test_await_same_aclose(self) -> None:
        it = agen_copy(1, 2)
        nxt = it.aclose()
        await nxt
        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            await nxt

    def test_throw_same_aclose(self) -> None:
        it = agen_copy(1, 2)
        nxt = it.aclose()
        with pytest.raises(StopIteration):
            nxt.throw(GeneratorExit)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            nxt.throw(GeneratorExit)

    def test_throw_custom_same_aclose(self) -> None:
        it = agen_copy(1, 2)
        nxt = it.aclose()
        with pytest.raises(MyError):
            nxt.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            nxt.throw(MyError)

    def test_throw_custom_same_athrow(self) -> None:
        it = agen_copy(1, 2)
        nxt = it.athrow(MyError)
        with pytest.raises(MyError):
            nxt.throw(MyError)

        with pytest.raises(
            RuntimeError,
            match=r'cannot reuse already awaited aclose\(\)/athrow\(\)',
        ):
            nxt.throw(MyError)

    @pytest.mark.asyncio
    @decorate
    async def test_aclose_with_different_coros(self) -> None:
        # Regression test for https://bugs.python.org/issue39606
        it = agen_copy(1, 2)
        await it.aclose()
        await it.aclose()


class TestAsyncGenAclose:
    @pytest.mark.asyncio
    @decorate
    async def test_after_exhaustion(self) -> None:
        # Regression test for https://bugs.python.org/issue39606
        it = agen_copy(1, 2)
        async for _ in it:
            pass
        await it.aclose()

    @pytest.mark.asyncio
    @decorate
    async def test_compatible_with_get_stack(self) -> None:
        ag = agen_copy(object())
        asyncio.create_task(ag.aclose())  # noqa: RUF006
        tasks = asyncio.all_tasks()
        for task in tasks:
            # No AttributeError raised
            task.get_stack()

    def test_throw(self) -> None:
        ag = agen_copy()
        with pytest.raises(MyError):
            ag.aclose().throw(MyError)

        del ag
        gc_collect()  # does not warn unawaited


class TestAsyncGenUnused:
    def test_asend(self) -> None:
        # gh-113753: asend objects allocated from a free-list should warn.
        # Ensure there is a finalized 'asend' object ready to be reused.
        try:
            ag = agen_copy(1, 2)
            ag.asend(None).send(None)
        except StopIteration:
            pass

        ag = agen_copy()
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'asend' of '.*' was never awaited",
        ):
            ag.asend(None)  # type: ignore[unused-coroutine]

    def test_athrow(self) -> None:
        ag = agen_copy(1, 2)
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'athrow' of '.*' was never awaited",
        ):
            ag.athrow(RuntimeError)  # type: ignore[unused-coroutine]

    def test_aclose(self) -> None:
        ag = agen_copy(1, 2)
        with pytest.warns(
            RuntimeWarning,
            match="coroutine method 'aclose' of '.*' was never awaited",
        ):
            ag.aclose()  # type: ignore[unused-coroutine]


class TestAlreadyRunning:
    def test_asend_send(self) -> None:
        ag = old_agenfn_inf_aw()
        g = ag.asend(None)
        g.send(None)
        g2 = ag.asend(None)

        with pytest.raises(
            RuntimeError,
            match=r'anext\(\): asynchronous generator is already running',
        ):
            g2.send(None)

        del g2
        gc_collect()  # does not warn unawaited

    def test_athrow_send(self) -> None:
        ag = old_agenfn_inf_aw()
        g = ag.asend(None)
        g.send(None)
        g2 = ag.athrow(Exception)

        with pytest.raises(
            RuntimeError,
            match=r'athrow\(\): asynchronous generator is already running',
        ):
            g2.send(None)

        del g2
        gc_collect()  # does not warn unawaited
