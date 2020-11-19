__all__ = ['buffered']

import contextlib
import enum
import queue
import threading
from concurrent import futures
from typing import TypeVar, Union

from .len_helpers import MaybeSizedIterable, MaybeSizedIterator, as_sized


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_MaybeEmpty = Union[_T, _Empty]
_empty = _Empty.token


@as_sized(hint=lambda it, _: len(it))
def buffered(iterable: MaybeSizedIterable[_T],
             latency: int = 2) -> MaybeSizedIterator[_T]:
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """
    q: 'queue.Queue[_MaybeEmpty[_T]]' = queue.Queue(latency)
    stop = threading.Event()

    def consume():
        with contextlib.ExitStack() as push:
            push.callback(q.put, _empty)  # match last q.get
            push.callback(q.put, _empty)  # signal to stop iteration
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item)
            if stop.is_set():
                push.pop_all()

    with contextlib.ExitStack() as pull:
        src = pull.enter_context(futures.ThreadPoolExecutor(1))
        pull.callback(q.get)  # unlock q.put in consume on Exception in main
        pull.callback(stop.set)

        task = src.submit(consume)
        while (item := q.get()) is not _empty:
            yield item
        task.result()  # throws if `consume` is dead
