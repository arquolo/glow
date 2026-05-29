import gc
import inspect
import sys
import weakref
from collections.abc import Generator
from types import GeneratorType
from typing import Any, Literal, Never, NoReturn, Self, cast

import pytest

from glow import time_this

decorate = time_this()


class MyError(Exception):
    pass


class BadBaseError(Exception):
    def __new__(cls, *args, **kwargs) -> type[Self]:  # type: ignore[misc]
        return cls


def gc_collect() -> None:
    gc.collect()
    gc.collect()
    gc.collect()


def throw(exc: type[Exception]) -> NoReturn:
    raise exc


# ---------------------------- generator mixtures ----------------------------


def as_gen[Y, S, R](obj: Generator[Y, S, R]) -> GeneratorType[Y, S, R]:
    return cast('GeneratorType[Y, S, R]', obj)


@overload
def _gen_once() -> Generator[None, Any]: ...
@overload
def _gen_once[T](v: T, /) -> Generator[T, Any]: ...
@decorate
def _gen_once[T](v: T = None, /) -> Generator[T, Any]:
    yield v


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
