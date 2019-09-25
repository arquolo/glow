__all__ = ('buffered', 'detach', 'mapped')

import os
import pickle
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread

from ..decos import Timed, close_at_exit
from .more import chunked, eat, iter_none


class Worker:
    """Lazy worker. Shares state of `fn` between subprocesses"""
    _manager = None
    _fn = None

    def __init__(self, fn):
        if self._manager is None:
            type(self)._manager = Timed(factory=Manager, timeout=10)
        manager = self._manager.get()
        self._shared = manager.Value('c', pickle.dumps(fn, protocol=-1))
        # ? maybe use:
        # (self._shared := manager.namespace).fn = fn

    def __call__(self, *args_tuple):
        if self._fn is None:
            type(self)._fn = staticmethod(pickle.loads(self._shared.value))
        return tuple(self._fn(*args) for args in args_tuple)


@close_at_exit
def buffered(iterable, latency=2, cleanup=None):
    """
    Moves iteration over iterable to another thread. Returns new iterable.

    Parameters
    ----------
        latency
            count of items can go ahead, by default 2.
        cleanup
            callback to apply for each `item` if failure happens
    """
    q = Queue(latency)
    stop = Event()

    def consume():
        with ExitStack() as push:
            push.callback(q.put, None)
            push.callback(q.put, None)
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item)

    with ExitStack() as pull:
        src = pull.enter_context(ThreadPoolExecutor(1))
        if cleanup is not None:  # maybe useless
            pull.callback(lambda: eat(cleanup(x) for x in iter_none(q.get)))
        pull.callback(stop.set)

        task = src.submit(consume)
        yield from iter_none(q.get)
        task.result()  # throws if `consume` is dead


@close_at_exit
def mapped(fn, *iterables, workers=None, latency=2, offload=0):
    """
    Concurrently applies `fn` callable to each element in zipped `iterables`.
    Keeps order. Never hang. Friendly to CTRL+C.

    Parameters
    ----------
        workers
            count of workers, by default `os.cpu_count()`
        latency
            count of tasks each workers can grab, by default 2.
        offload
            if not zero enables usage of `Process` instead of `Thread`,
            number means chunk size for each Process, by default 0
    """
    workers = workers or os.cpu_count()

    with ExitStack() as stack:
        iterable = zip(*iterables)
        if offload:
            fn = Worker(fn)
            pool = stack.enter_context(ProcessPoolExecutor(workers))
            iterable = chunked(iterable, size=offload)
        else:
            pool = stack.enter_context(ThreadPoolExecutor(workers))

        fs = (pool.submit(fn, *items) for items in iterable)
        fs = buffered(fs, latency=latency * workers, cleanup=Future.cancel)
        results = (future.result() for future in fs)

        if offload:
            results = (r for rs in results for r in rs)
        yield from results


def detach(iterable):
    Thread(target=eat, args=(iterable,), daemon=True).start()
