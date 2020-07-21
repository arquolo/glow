__all__ = ('buffered', )

import contextlib
import enum
import queue
import threading
from concurrent import futures
from typing import Callable, Iterable, Iterator, Union, TypeVar

from .len_helpers import as_sized
from .more import iter_none


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_MaybeEmpty = Union[_T, _Empty]


@as_sized(hint=lambda it, _: len(it))
def buffered(iterable: Iterable[_T], latency: int = 2) -> Iterator[_T]:
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """
    q: 'queue.Queue[_MaybeEmpty[_T]]' = queue.Queue(latency)
    stop = threading.Event()

    def consume():
        with contextlib.ExitStack() as push:
            push.callback(q.put, _Empty.token)  # match last q.get
            push.callback(q.put, _Empty.token)  # signal to iter_none
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item)
            if stop.is_set():
                push.pop_all()

    with contextlib.ExitStack() as pull:
        src = pull.enter_context(futures.ThreadPoolExecutor(1))
        pull.callback(q.get)  # unlock q.put in consume on Exception in main
        pull.callback(stop.set)

        task = src.submit(consume)
        q_get: Callable[[], _MaybeEmpty[_T]] = q.get
        yield from iter_none(q_get, _Empty.token)
        task.result()  # throws if `consume` is dead
