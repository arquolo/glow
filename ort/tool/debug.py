import inspect
import sys
from collections import Counter
from contextlib import contextmanager, suppress
from threading import RLock
from time import time
from types import ModuleType

from wrapt import (decorator, register_post_import_hook, synchronized,
                   ObjectProxy)

from . import export


@export
@synchronized
def prints(*args, **kwargs):
    print(*args, **kwargs)


@export
@contextmanager
def timer(name='Task'):
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        prints(f'{name} done in {duration:.4g} seconds')


@export
@decorator
def profile(func, _, args, kwargs):
    with timer(f'{func.__module__}:{func.__qualname__}'):
        return func(*args, **kwargs)

#############################################################################


def strip_dirs(path,
               _paths=tuple(sorted(sys.path, key=len, reverse=True))):
    for prefix in _paths:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def whereami():
    names = (f'{strip_dirs(f.filename) or ""}:{f.function}:{f.lineno}'
             for f in reversed(inspect.stack())
             if 'importlib' not in f.filename
                and f.function not in ('whereami', 'trace'))
    return ' -> '.join(names)


@export
@decorator
def trace(func, _, args, kwargs):
    prints(f'<({whereami()})> : {func.__module__ or ""}.{func.__qualname__}',
           flush=True)
    return func(*args, **kwargs)


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
            decorated.__module__ = (getattr(decorated, '__module__', '') or
                                    getattr(member, '__module__', '') or
                                    getattr(obj, '__module__', '') or
                                    getattr(module, '__name__', ''))
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


@export
def trace_module(name):
    register_post_import_hook(set_trace, name)

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


@export
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
