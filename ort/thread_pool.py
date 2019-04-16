import contextlib
import os
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Thread, get_ident


# TODO: too complex, needs refactoring
class ContextQueue(Queue, Future, contextlib.ContextDecorator):

    def __init__(self, maxsize=0, timeout=.001):
        Queue.__init__(self, maxsize=maxsize)
        Future.__init__(self)

        self._timeout = timeout
        self._id = None

    def __enter__(self):
        if not self._id:
            self._id = get_ident()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.done():
            if not exc_type:
                return False
            self.set_exception(exc_type(exc_value).with_traceback(traceback))
        if self.done() and self._id == get_ident():
            raise self.exception() from None
        return True

    def put(self, item):  # pylint: disable=arguments-differ
        while not self.done():
            with contextlib.suppress(Full):
                return super().put(item, timeout=self._timeout)
        return True

    def get(self):  # pylint: disable=arguments-differ
        while not self.done():
            with contextlib.suppress(Empty):
                return super().get(timeout=self._timeout)

    def bufferize(self, iterable):
        def produce():
            for item in iterable:
                if self.put(item):
                    return
            self.put(None)

        Thread(target=self(produce), name='Buffer').start()
        yield from iter(self.get, None)


def bufferize(iterable, count=1):
    with ContextQueue(count) as q:
        yield from q.bufferize(iterable)


def maps(function, iterable, prefetch=0, workers=os.cpu_count()):
    """Lazy, exception-safe, buffered and concurrent `builtins.map`"""
    with ThreadPoolExecutor(workers, function.__qualname__) as executor:
        with ContextQueue(prefetch) as q:
            for future in q.bufferize(executor.submit(function, item)
                                      for item in iterable):
                yield future.result()
