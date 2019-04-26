import functools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from queue import Queue
from threading import Event, Thread

_func = None


def worker(state, item):
    global _func  # pylint: disable=global-statement
    if _func is None:
        _func = pickle.loads(state.value)
    return _func(item)


def bufferize(iterable, latency=2, cleanup=None):
    q = Queue(latency)
    stop = Event()

    def consume():
        for item, _ in zip(iterable, iter(stop.is_set, True)):
            q.put(item)

    with ThreadPoolExecutor(1, 'src') as src:
        try:
            task = src.submit(consume)
            task.add_done_callback(lambda _: [q.put(None) for _ in range(2)])
            yield from iter(q.get, None)

        except:  # pylint: disable=bare-except
            if not (task.done() and task.exception()):
                stop.set()  # exception came from callback, terminate src thread
                raise

        finally:
            if cleanup is not None:
                for item in iter(q.get, None):
                    cleanup(item)
            exc = task.exception()
            if exc:
                raise exc from None  # rethrow source exception


def maps(func, *iterables, workers=None, latency=2, offload=False):
    """Lazy, exception-safe, buffered and concurrent `builtins.map`"""
    workers = workers or os.cpu_count()
    if offload:  # put function to shared memory
        func = functools.partial(
            worker,
            Manager().Value('c', pickle.dumps(func))
        )

    _Pool = ProcessPoolExecutor if offload else ThreadPoolExecutor
    with _Pool(workers) as pool:
        for f in bufferize(
            (pool.submit(func, *items) for items in zip(*iterables)),
            latency=latency * workers,
            cleanup=lambda f: f.cancel()
        ):
            yield f.result()


def map_detach(func, iterable, workers=None, latency=2, offload=False):
    def fetch():
        for _ in maps(func, iterable,
                      workers=workers, latency=latency, offload=offload):
            pass

    Thread(target=fetch, daemon=True).start()
