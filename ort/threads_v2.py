import functools
import inspect
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager
from queue import Queue
from threading import Event

_func = None

def worker(state, item):
    global _func
    if _func is None:
        _func = pickle.loads(state.value)
    return _func(item)
     

def map_t(func, iterable,
          workers=os.cpu_count(), latency=2, offload=False):
    def submit():
        pool = ProcessPoolExecutor if offload else ThreadPoolExecutor
        with pool(workers) as pool:
            for _, item in zip(iter(e.is_set, True), iterable):
                q.put(pool.submit(func, item))  # stops when main dies

    if offload and not inspect.isfunction(func):
        func = functools.partial(
            worker,
            Manager().Value('c', pickle.dumps(func)),
        )
    q = Queue(workers * latency)
    e = Event()

    with ThreadPoolExecutor(1) as server:
        task = server.submit(submit)
        task.add_done_callback(lambda _: [q.put(None) for _ in range(2)])

        try:
            source_exc = None
            for f in iter(q.get, None):  # breaks if source dead or exhausted
                yield f.result()  # throws if func fails
            if task.done():
                source_exc = task.exception()

        except:  # pylint: disable=bare-except
            if not source_exc:
                e.set()  # exception came from f.result()
                raise

        finally:
            for f in iter(q.get, None):
                f.cancel()
            if source_exc:
                raise source_exc from None  # rethrow source exception


def map_in_background(func, it, workers=12, latency=2):
    def fetch():
        for _ in map_t(func, it, workers=workers, latency=latency):
            pass

    with ThreadPoolExecutor(1) as pool:
        pool.submit(fetch)
