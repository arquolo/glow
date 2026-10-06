import asyncio
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from typing import Any, NoReturn, Self

import pytest

from glow._wrap import wrap


class Recorder:
    def __init__(self) -> None:
        self.now = 0
        self.wait = 0
        self.suspensions = 0

    def new_call(self) -> None:
        pass

    def __call__[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return fn(*args, **kwargs)

    def suspend(self) -> Callable[[], None]:
        start = self.now

        def resume() -> None:
            self.wait += self.now - start
            self.suspensions += 1

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


def await_iterator[I: Iterator](iterator: I, recorder: Recorder) -> I:
    return wrap(lambda: AwaitableIterator(iterator), recorder)().__await__()


def test_await_iterator_records_suspension_only_on_resume() -> None:
    recorder = Recorder()
    iterator = await_iterator(PlainIterator(), recorder)
    next(iterator)
    recorder.now += 7
    assert (recorder.suspensions, recorder.wait) == (0, 0)

    next(iterator)
    assert (recorder.suspensions, recorder.wait) == (1, 7)

    recorder.now += 100
    del iterator
    # Discarding an iterator without resuming adds neither time nor an event.
    assert (recorder.suspensions, recorder.wait) == (1, 7)


def test_await_iterator_preserves_missing_optional_methods() -> None:
    recorder = Recorder()

    class FiniteIterator(PlainIterator):
        remaining = 2

        def __next__(self) -> None:
            if not self.remaining:
                raise StopIteration(42)
            self.remaining -= 1

    iterator = await_iterator(FiniteIterator(), recorder)
    recorder.now = 100  # Waiting before the first step is not suspension.
    assert next(iterator) is None
    recorder.now += 3
    for name in ('send', 'throw', 'close'):
        assert not hasattr(iterator, name)
    # Looking up a missing method does not resume the underlying operation.
    assert (recorder.suspensions, recorder.wait) == (0, 0)
    assert next(iterator) is None
    recorder.now += 5
    with pytest.raises(StopIteration) as stopped:
        next(iterator)
    assert stopped.value.value == 42
    recorder.now += 100
    assert (recorder.suspensions, recorder.wait) == (2, 8)


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
        recorded = (recorder.suspensions, recorder.wait)
        recorder.now += 7
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # No throw/close hook resumes this iterator on cancellation.
        assert (recorder.suspensions, recorder.wait) == recorded
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.parametrize('method', ['send', 'throw'])
def test_await_iterator_optional_method_records_only_resumed_intervals(
    method,
) -> None:
    value = object()
    error = ValueError('original')
    received = []

    # Deliberately omit close: this is an iterator, not a full Generator.
    class PartialIterator(PlainIterator):
        done = False

        def __next__(self) -> None:
            if self.done:
                raise StopIteration

        def send(self, item) -> object:
            self.done = True
            received.append(item)
            return value

        def throw(self, exc) -> object:
            self.done = True
            received.append(exc)
            return value

    recorder = Recorder()
    iterator = await_iterator(PartialIterator(), recorder)
    next(iterator)
    recorder.now += 3
    argument = value if method == 'send' else error
    assert getattr(iterator, method)(argument) is value
    assert len(received) == 1
    assert received[0] is argument
    assert (recorder.suspensions, recorder.wait) == (1, 3)
    recorder.now += 5
    assert not hasattr(iterator, 'close')
    with pytest.raises(StopIteration):
        next(iterator)
    assert (recorder.suspensions, recorder.wait) == (2, 8)


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
    assert (recorder.suspensions, recorder.wait) == (1, 4)


@pytest.mark.parametrize('method', ['send', 'throw'])
def test_await_iterator_failed_method_records_previous_interval(
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
    assert (recorder.suspensions, recorder.wait) == (1, 3)


class ExecutionRecorder(Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.executing = 0

    def __call__[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        start = self.now
        try:
            return fn(*args, **kwargs)
        finally:
            self.executing += self.now - start


def test_generator_counts_steps_but_not_consumer_time() -> None:
    recorder = ExecutionRecorder()
    payload = iter([1])

    def generate() -> Generator[Iterator[int], Any]:
        recorder.now += 3
        yield payload
        recorder.now += 5

    def factory() -> Generator[Iterator[int], Any]:
        recorder.now += 2
        return generate()

    iterator = wrap(factory, recorder)()
    recorder.now += 100
    assert next(iterator) is payload
    recorder.now += 100
    assert list(payload) == [1]
    with pytest.raises(StopIteration):
        next(iterator)
    assert recorder.executing == 10
    assert (recorder.suspensions, recorder.wait) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
async def test_async_iteration_counts_internal_wait_not_consumer_time(native):
    recorder = ExecutionRecorder()
    payload = iter([1])

    async def step() -> Iterator[int]:
        recorder.now += 2
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: setattr(recorder, 'now', recorder.now + 3))
        await asyncio.sleep(0)
        recorder.now += 5
        return payload

    async def generate() -> AsyncGenerator[Iterator[int], Any]:
        yield await step()

    class CustomIterator:
        def __init__(self) -> None:
            self.done = False

        def __aiter__(self) -> Self:
            return self

        async def __anext__(self) -> Iterator[int]:
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return await step()

    iterator = wrap(generate if native else CustomIterator, recorder)()
    recorder.now += 100
    assert await anext(iterator) is payload
    recorder.now += 100
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)
    assert recorder.executing == 7
    assert (recorder.suspensions, recorder.wait) == (1, 3)


class CustomCoroutine(Coroutine):
    def __init__(self, coroutine: Coroutine) -> None:
        self.coroutine = coroutine

    def __await__(self) -> Generator[Any, Any, Any]:
        return self.coroutine.__await__()

    def send(self, value) -> Any:
        return self.coroutine.send(value)

    def throw(self, *args) -> Any:
        return self.coroutine.throw(*args)

    def close(self) -> None:
        return self.coroutine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('as_task', [False, True])
async def test_coroutine_wait_and_result_identity(native, as_task) -> None:
    recorder = ExecutionRecorder()
    result = iter([1])

    async def work() -> Iterator[int]:
        recorder.now += 2
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: setattr(recorder, 'now', recorder.now + 3))
        await asyncio.sleep(0)
        recorder.now += 5
        return result

    def factory() -> Coroutine[Any, Any, Iterator[int]] | CustomCoroutine:
        return work() if native else CustomCoroutine(work())

    coroutine = wrap(factory, recorder)()
    recorder.now += 100
    actual = await (asyncio.create_task(coroutine) if as_task else coroutine)
    assert actual is result
    recorder.now += 100
    assert list(actual) == [1]
    assert recorder.executing == 7
    assert (recorder.suspensions, recorder.wait) == (1, 3)


@pytest.mark.asyncio
async def test_awaitable_returning_itself_as_iterator() -> None:
    recorder = Recorder()

    class SelfAwaitable:
        def __await__(self) -> Self:
            return self

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> NoReturn:
            raise StopIteration(42)

    assert await wrap(SelfAwaitable, recorder)() == 42
    assert recorder.suspensions == 0
