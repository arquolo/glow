import contextlib
import functools
import inspect
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread

_func = None


def worker(state, item):
    global _func  # pylint: disable=global-statement
    if _func is None:
        _func = pickle.loads(state.value)
    return _func(item)


def maps(func, iterable, workers=None, latency=2, offload=False):
    if func is None:
        workers = 1
        pool = contextlib.nullcontext()  # bufferize
    else:
        workers = workers or os.cpu_count()
        if offload:
            pool = ProcessPoolExecutor(workers)
            if not inspect.isfunction(func):  # put function to shared memory
                func = functools.partial(
                    worker,
                    Manager().Value('c', pickle.dumps(func))
                )
        else:
            pool = ThreadPoolExecutor(workers)

    q = Queue(workers * latency)
    stop = Event()

    def submit():
        with pool:  # stops when main dies
            for item, _ in zip(iterable, iter(stop.is_set, True)):
                q.put(item if func is None else pool.submit(func, item))

    with ThreadPoolExecutor(1, thread_name_prefix='src') as src:
        try:
            task = src.submit(submit)
            task.add_done_callback(lambda _: [q.put(None) for _ in range(2)])
            for f in iter(q.get, None):
                yield f if func is None else f.result()

        except:
            stop.set()  # exception came from func, terminate src thread
            raise

        finally:
            if func is not None:
                for f in iter(q.get, None):
                    f.cancel()
            exc = task.exception()
            if exc:
                raise exc from None  # rethrow source exception


def bufferize(iterable, latency=2):
    yield from maps(None, iterable, latency=latency, offload=False)


def map_detach(func, iterable, workers=None, latency=2, offload=False):
    def fetch():
        for _ in maps(func, iterable,
                      workers=workers, latency=latency, offload=offload):
            pass

    Thread(target=fetch, daemon=True).start()
