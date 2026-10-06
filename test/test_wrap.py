import asyncio
import warnings
from collections.abc import Callable, Generator, Iterator
from typing import Any, NoReturn, Self

import pytest

from glow._wrap import wrap


class Recorder:
    def __init__(self) -> None:
        self.now = 0
        self.wait = 0
        self.started = 0
        self.finished = 0

    def new_call(self) -> None:
        pass

    def __call__[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return fn(*args, **kwargs)

    def suspend(self) -> Callable[[], None]:
        start = self.now
        self.started += 1

        def resume() -> None:
            self.wait += self.now - start
            self.finished += 1

        return resume


class PlainIterator:
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> None:
        return None


class AwaitableIterator[T]:
    def __init__(self, iterator: Iterator[T]) -> None:
        self.iterator = iterator

    def __await__(self) -> Iterator[T]:
        return self.iterator


def await_iterator[T](
    iterator: Iterator[T], recorder: Recorder
) -> Iterator[T]:
    return wrap(lambda: AwaitableIterator(iterator), recorder)().__await__()


def test_await_iterator_send_none_and_close_without_optional_methods() -> None:
    recorder = Recorder()
    iterator = await_iterator(PlainIterator(), recorder)
    recorder.now = 100  # Waiting before the first step is not suspension.
    assert next(iterator) is None
    recorder.now += 3
    assert iterator.send(None) is None
    recorder.now += 5
    assert iterator.close() is None
    recorder.now += 100
    iterator.close()  # Closing again must not account for another interval.
    assert (recorder.started, recorder.finished, recorder.wait) == (2, 2, 8)


@pytest.mark.asyncio
async def test_await_iterator_cancellation_without_throw() -> None:
    recorder = Recorder()
    entered = asyncio.Event()

    class IteratorWithSignal(PlainIterator):
        def __next__(self) -> None:
            entered.set()

    async def run() -> None:
        await wrap(lambda: AwaitableIterator(IteratorWithSignal()), recorder)()

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        recorder.now += 7
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recorder.started == recorder.finished
        assert recorder.wait == 7
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.parametrize('method', ['send', 'throw'])
def test_await_iterator_optional_method_starts_new_suspension(method) -> None:
    value = object()
    error = ValueError('original')
    received = []

    # Deliberately omit close: this is an iterator, not a full Generator.
    class PartialIterator(PlainIterator):
        def send(self, item) -> object:
            received.append(item)
            return value

        def throw(self, exc) -> object:
            received.append(exc)
            return value

    recorder = Recorder()
    iterator = await_iterator(PartialIterator(), recorder)
    next(iterator)
    recorder.now += 3
    argument = value if method == 'send' else error
    assert getattr(iterator, method)(argument) is value
    assert received == [argument]
    assert received[0] is argument
    assert (recorder.started, recorder.finished) == (2, 1)
    recorder.now += 5
    iterator.close()
    assert (recorder.started, recorder.finished, recorder.wait) == (2, 2, 8)


def test_await_iterator_close_delegates_and_preserves_result() -> None:
    result = object()
    calls = []

    class CloseIterator(PlainIterator):
        def close(self) -> object:
            calls.append('close')
            return result

    recorder = Recorder()
    iterator = await_iterator(CloseIterator(), recorder)
    next(iterator)
    recorder.now += 4
    assert iterator.close() is result
    assert calls == ['close']
    assert (recorder.started, recorder.finished, recorder.wait) == (1, 1, 4)


@pytest.mark.parametrize('method', ['send', 'throw'])
def test_await_iterator_failed_method_does_not_start_suspension(
    method,
) -> None:
    error = ValueError('original')

    class FailingIterator(PlainIterator):
        def send(self, value) -> NoReturn:
            raise error

        def throw(self, exc) -> NoReturn:
            raise error

    recorder = Recorder()
    iterator = await_iterator(FailingIterator(), recorder)
    next(iterator)
    recorder.now += 3
    with pytest.raises(ValueError) as caught:
        getattr(iterator, method)(error)
    assert caught.value is error
    assert (recorder.started, recorder.finished, recorder.wait) == (1, 1, 3)


@pytest.mark.parametrize(
    'args',
    [
        (ValueError,),
        (ValueError, ('a', 'b')),
        (ValueError, TypeError('nested')),
        (ValueError('original'),),
        (StopIteration(42),),
        (asyncio.CancelledError('cancelled'),),
        (ValueError('original'), 'invalid separate value'),
    ],
)
def test_await_iterator_throw_without_method_matches_generator(args) -> None:
    def empty() -> Generator[None, Any]:
        yield

    # Multi-argument throw is deprecated but still supported by Python.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(BaseException) as expected:
            empty().throw(*args)

        recorder = Recorder()
        iterator = await_iterator(PlainIterator(), recorder)
        next(iterator)
        recorder.now += 6
        with pytest.raises(BaseException) as actual:
            iterator.throw(*args)

    assert type(actual.value) is type(expected.value)
    assert actual.value.args == expected.value.args
    if len(args) == 1 and isinstance(args[0], BaseException):
        assert actual.value is args[0]
    assert (recorder.started, recorder.finished, recorder.wait) == (1, 1, 6)
