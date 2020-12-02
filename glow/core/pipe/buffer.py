__all__ = ['buffered']

import atexit
import enum
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from queue import Queue
from threading import Event
from typing import (Callable, Iterable, Iterator, Optional, Protocol, Tuple,
                    TypeVar, Union)

import loky
import loky.backend


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_MaybeEmpty = Union[_T, _Empty]
_empty = _Empty.token


class _IQueue(Protocol[_T]):
    def get(self) -> _T:
        ...

    def put(self, value: _T) -> None:
        ...


class _IEvent(Protocol):
    def is_set(self) -> bool:
        ...

    def set(self) -> None:  # noqa: A003
        ...


def _setup(
    stack: ExitStack,
    latency: int,
    num_workers: int,
) -> Tuple[Executor, _IQueue, _IEvent, Optional[Callable]]:
    ex: Executor
    if not num_workers:
        return stack.enter_context(
            ThreadPoolExecutor(1)), Queue(latency), Event(), None

    context = loky.backend.context.get_context('loky_init_main')
    ex = loky.get_reusable_executor(num_workers, context)  # type: ignore
    mgr = stack.enter_context(context.Manager())
    return (ex, mgr.Queue(latency), mgr.Event(),
            atexit.register(ex.shutdown, kill_workers=True))


@dataclass
class _Buffered(Iterable[_T]):
    iterable: Iterable[_T]
    latency: int = 2
    num_workers: int = 0

    def _consume(self, q: _IQueue[_MaybeEmpty[_T]], stop: _IEvent):
        with ExitStack() as stack:
            stack.callback(q.put, _empty)  # Match last q.get
            stack.callback(q.put, _empty)  # Signal to stop iteration

            for item, _ in zip(self.iterable, iter(stop.is_set, True)):
                q.put(item)

            # if stop.is_set():
            #     stack.pop_all()
            # q.put(_empty)  # Will happen if all ok to notify main to stop

    def __iter__(self) -> Iterator[_T]:
        with ExitStack() as stack:
            q: _IQueue[_MaybeEmpty[_T]]
            ex, q, stop, terminator = _setup(stack, self.latency,
                                             self.num_workers)

            stack.callback(q.get)  # Wakes q.put when main fails
            stack.callback(stop.set)

            task = ex.submit(self._consume, q, stop)
            while (item := q.get()) is not _empty:
                yield item
            task.result()  # Throws if `consume` is dead

        if terminator is not None:  # Cancel killing workers if all ok
            atexit.unregister(terminator)

    def __len__(self) -> int:
        return len(self.iterable)  # type: ignore


def buffered(iterable: Iterable[_T],
             latency: int = 2,
             num_workers: int = 0) -> _Buffered[_T]:
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """
    return _Buffered(iterable, latency, num_workers)
