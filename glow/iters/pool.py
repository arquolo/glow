__all__ = 'buffered', 'detach', 'mapped'

import collections
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread

from ..decos import Timed
from .more import iter_none


class _SharedWorker:
    """Lazy worker. Shares state of `fn` between subprocesses"""
    _manager = None
    _fn = None

    def __init__(self, fn):
        if self._manager is None:
            type(self)._manager = Timed(factory=Manager, timeout=10)
        manager = self._manager.get()
        self._shared = manager.Value('c', pickle.dumps(fn, protocol=-1))

    def __call__(self, *args):
        if self._fn is None:
            type(self)._fn = staticmethod(pickle.loads(self._shared.value))
        return self._fn(*args)


def buffered(iterable, latency=2, cleanup=None):
    q = Queue(latency)
    stop = Event()

    def consume():
        with ExitStack() as push:
            push.callback(q.put, None)
            push.callback(q.put, None)
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item)

    with ThreadPoolExecutor(1, 'src') as src:
        with ExitStack() as pull:
            if cleanup is not None:
                pull.callback(lambda: {cleanup(f) for f in iter_none(q.get)})
            pull.callback(stop.set)

            task = src.submit(consume)
            yield from iter_none(q.get)
            task.result()  # if `consume` dies, never called


def mapped(fn, *iterables, workers=None, latency=2, offload=False):
    """Lazy, exception-safe, buffered and concurrent `builtins.map`"""
    workers = workers or os.cpu_count()
    if offload:
        fn = _SharedWorker(fn)
        pool = ProcessPoolExecutor(workers)
    else:
        pool = ThreadPoolExecutor(workers)

    with pool:
        for f in buffered(
            (pool.submit(fn, *items) for items in zip(*iterables)),
            latency=latency * workers,
            cleanup=lambda f: f.cancel(),
        ):
            yield f.result()


def detach(iterator):
    Thread(target=collections.deque, args=(iterator,), kwargs={'maxlen': 0},
           daemon=True).start()
