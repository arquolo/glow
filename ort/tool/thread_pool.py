import collections
import functools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread

from . import export, iter_none

_func = None


def worker(state, item):
    global _func
    if _func is None:
        _func = pickle.loads(state.value)
    return _func(item)


@export
def buffered(iterable, latency=2, cleanup=None):
    q = Queue(latency)
    stop = Event()

    def consume():
        for item, _ in zip(iterable, iter(stop.is_set, True)):
            q.put(item)

    with ThreadPoolExecutor(1, 'src') as src:
        try:
            task = src.submit(consume)
            task.add_done_callback(lambda _: [q.put(None) for _ in range(2)])
            yield from iter_none(q.get)

        except BaseException:
            if not (task.done() and task.exception()):
                stop.set()  # exception came from callback, terminate src
                raise

        finally:
            if cleanup is not None:
                for item in iter_none(q.get):
                    cleanup(item)
            exc = task.exception()
            if exc:
                raise exc from None  # rethrow source exception


@export
def mapped(fn, *iterables, workers=None, latency=2, offload=False):
    """Lazy, exception-safe, buffered and concurrent `builtins.map`"""
    workers = workers or os.cpu_count()
    if offload:  # put function to shared memory
        fn = functools.partial(
            worker,
            Manager().Value('c', pickle.dumps(fn))
        )

    _pool = ProcessPoolExecutor if offload else ThreadPoolExecutor
    with _pool(workers) as pool:
        for f in buffered(
            (pool.submit(fn, *items) for items in zip(*iterables)),
            latency=latency * workers,
            cleanup=lambda f: f.cancel()
        ):
            yield f.result()


@export
def detach(iterator):
    Thread(target=collections.deque, args=(iterator,), kwargs={'maxlen': 0},
           daemon=True).start()
