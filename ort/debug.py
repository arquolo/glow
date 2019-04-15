import inspect
import logging
import sys
from collections import Counter
from contextlib import contextmanager, suppress
from threading import RLock
from time import time
from types import ModuleType

from wrapt import (decorator, when_imported, synchronized,
                   ObjectProxy)

sprint = synchronized(print)


@contextmanager
def timer(event_name=''):
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        logging.warning('%s - done in %.4g seconds', event_name, duration)


@decorator
def profile(wrapped, _, args, kwargs):
    with timer(f'{wrapped.__module__}:{wrapped.__qualname__}'):
        result = wrapped(*args, **kwargs)
    return result

#############################################################################


_PATH = tuple(sorted(sys.path, key=lambda path: -len(path)))


def clean_module_path(path):
    for prefix in _PATH:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def whereami():
    names = [':'.join([clean_module_path(frame.filename) or "",
                       frame.function,
                       str(frame.lineno)])
             for frame in inspect.stack()
             if (frame.function not in {'whereami', 'trace'}
                 and frame.filename.find('importlib') == -1)]
    return ' -> '.join(reversed(names))


@decorator
def trace(wrapped, _, args, kwargs):
    sprint('<(%s)> : %s.%s' % (whereami(),
                               wrapped.__module__ or "", wrapped.__qualname__),
           flush=True)
    return wrapped(*args, **kwargs)


def set_trace(obj, seen=None, prefix=None, module=None):
    if isinstance(obj, ModuleType):
        if seen is None:
            seen = set()
            prefix = obj.__name__
        if not obj.__name__.startswith(prefix) or obj.__name__ in seen:
            return
        seen.add(obj.__name__)
        for name in dir(obj):
            set_trace(getattr(obj, name), module=obj, seen=seen, prefix=prefix)

    if not callable(obj):
        return

    if not hasattr(obj, '__dict__'):
        setattr(module, obj.__qualname__, trace(obj))  # pylint: disable=no-value-for-parameter
        print(f'wraps "{module.__name__}:{obj.__qualname__}"')
        return

    for name in obj.__dict__:
        with suppress(AttributeError, TypeError):
            member = getattr(obj, name)
            if not callable(member):
                continue
            decorated = trace(member)  # pylint: disable=no-value-for-parameter
            name_ = (getattr(decorated, '__module__', '') or
                     getattr(member, '__module__', '') or
                     getattr(obj, '__module__', '') or
                     getattr(module, '__name__', ''))
            setattr(decorated, '__module__', name_)
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


def trace_module(name):
    return when_imported(name)(set_trace)

#############################################################################


@decorator
def threadsafe_coroutine(wrapped, _, args, kwargs):
    coro = wrapped(*args, **kwargs)
    next(coro)
    lock = RLock()

    class Synchronized(ObjectProxy):  # pylint: disable=abstract-method
        def send(self, item):
            with lock:
                return self.__wrapped__.send(item)
        def __next__(self):
            return self.send(None)
    return Synchronized(coro)


@threadsafe_coroutine
def summary():
    state = Counter()
    while True:
        key = yield
        if key is None:
            state.clear()
            continue
        state[key] += 1
        print(dict(state), flush=True, end='\r')
