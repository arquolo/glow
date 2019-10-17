__all__ = ('buffered', 'detach', 'mapped')

import os
import signal
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import islice
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread
from typing import Callable, Iterable, Iterator, Optional, TypeVar

import loky

from ..core import sizeof
from ..decos import Reusable, close_at_exit
from .more import chunked, eat, iter_none
from .size_hint import make_sized

T = TypeVar('T')
loky.set_loky_pickler('pickle')
loky.backend.context.set_start_method('loky_init_main')


class Chunker:
    """Applies `fn` to chunk"""
    __slots__ = ('_fn',)

    def __init__(self, fn):
        self._fn = fn

    @property
    def fn(self) -> Callable:
        return self._fn

    def __call__(self, *args_tuple):
        fn = self.fn
        return tuple(fn(*args) for args in args_tuple)


class ChunkerShared(Chunker):
    """Applies `fn` to chunk. Shares state of `fn` between subprocesses"""
    __slots__ = ('_shared', '_parent')
    _manager = None
    _saved_fn = None
    _saved_parent = 0

    def __init__(self, fn):
        if self._manager is None:
            type(self)._manager = Reusable(Manager, timeout=60)

        self._shared = self._manager.get().Namespace()
        self._shared.fn = fn
        self._parent = id(self)

    @property
    def fn(self) -> Callable:
        if (self._saved_fn is None or  # not initialized
                self._saved_parent != self._parent):  # or outdated
            # # read from namespace, keep id
            type(self)._saved_fn = staticmethod(self._shared.fn)
            type(self)._saved_parent = self._parent

            # suppresses KeyboardInterrupt in child processes
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        return self._saved_fn


@make_sized
@close_at_exit
def buffered(iterable: Iterable[T],
             latency: int = 2,
             finalize: Optional[Callable[[T], None]] = None) -> Iterator[T]:
    """Moves iteration over iterable to another thread. Returns new iterable.

    Parameters:
      - `latency` - count of items can go ahead
        (default: `2`)
      - `finalize` - callback to apply for each `item` if failure happens
        (default: `None`)
    """
    q = Queue(latency)
    stop = Event()
    marker = object()
    if finalize is None:
        def finalize(_):
            pass

    def consume():
        with ExitStack() as push:
            push.callback(q.put, marker)
            push.callback(q.put, marker)
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item)

    with ExitStack() as pull:
        src = pull.enter_context(ThreadPoolExecutor(1))
        pull.callback(eat, map(finalize, iter_none(q.get, marker)))
        pull.callback(stop.set)

        task = src.submit(consume)
        yield from iter_none(q.get, marker)
        task.result()  # throws if `consume` is dead


@make_sized
@close_at_exit
def mapped(fn: Callable[..., T], *iterables: Iterable,
           workers: Optional[int] = None,
           latency: int = 2,
           offload: int = 0) -> Iterator[T]:
    """
    Concurrently applies `fn` callable to each element in zipped `iterables`.
    Keeps order. Never hang. Friendly to CTRL+C. Uses all processing power.

    Parameters:
      - `workers` - count of workers
        (default: same as `os.cpu_count()`)
      - `latency` - count of tasks each workers can grab
        (default: `2`)
      - `offload` - size of chunk to pass to each `Process`, if not `0`
        (default: `0`)
    """
    if workers == 0:
        yield from map(fn, *iterables)
        return

    workers = workers or os.cpu_count()
    latency *= workers
    iterable = zip(*iterables)

    with ExitStack() as stack:
        if offload:
            pool = loky.get_reusable_executor(max_workers=workers, timeout=60)
            iterable = chunked(iterable, size=offload)
            fn = (Chunker if sizeof(fn) < 2e5 else ChunkerShared)(fn)
        else:
            pool = ThreadPoolExecutor(workers)
            pool = stack.enter_context(pool)

        fs = deque(maxlen=latency)
        stack.callback(eat, (fut.cancel() for fut in reversed(fs)))

        fs_submit = (pool.submit(fn, *items) for items in iterable)
        fs.extend(islice(fs_submit, latency))

        while fs:
            result = fs.popleft().result()
            fs.extend(islice(fs_submit, 1))
            yield from (result if offload else [result])


def detach(iterable: Iterable) -> None:
    """Consume `iterable` asynchronously"""
    Thread(target=eat, args=(iterable,), daemon=True).start()
