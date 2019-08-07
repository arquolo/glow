__all__ = 'print_', 'trace', 'trace_module', 'summary'

import functools
import inspect
import itertools
from collections import Counter
from contextlib import suppress
from threading import RLock
from types import ModuleType

from wrapt import (decorator, register_post_import_hook,
                   ObjectProxy)


class Printer:
    """Thread-safe version of `print` function. Supports `tqdm` module"""
    lock = RLock()
    tqdm = None

    @functools.wraps(print)
    def __call__(self, *args, **kwargs):
        with self.lock:
            print(*args, **kwargs)

    @functools.wraps(print)
    def __call_tqdm__(self, *args, sep=' ', flush=False, **kwargs):
        self.tqdm.write(sep.join(map(str, args)), **kwargs)

    @classmethod
    def patch(cls, tqdm: ModuleType):
        cls.tqdm = tqdm.tqdm
        cls.__call__ = cls.__call_tqdm__


print_ = Printer()
register_post_import_hook(Printer.patch, 'tqdm')


#############################################################################


def stack():
    frame = inspect.currentframe()
    while frame is not None:
        f = inspect.getframeinfo(frame)
        module_name = inspect.getmodulename(f.filename)
        yield f'{module_name}:{f.function}:{f.lineno}'
        frame = frame.f_back


def whereami(skip=2):
    return ' -> '.join(reversed(list(itertools.islice(stack(), skip, None))))


@decorator
def trace(fn, _, args, kwargs):
    print_(f'<({whereami(3)})> : {fn.__module__ or ""}.{fn.__qualname__}',
           flush=True)
    return fn(*args, **kwargs)


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
        setattr(module, obj.__qualname__, trace(obj))
        print(f'wraps "{module.__name__}:{obj.__qualname__}"')
        return

    for name in obj.__dict__:
        with suppress(AttributeError, TypeError):
            member = getattr(obj, name)
            if not callable(member):
                continue
            decorated = trace(member)

            for m in (decorated, member, obj):
                with suppress(AttributeError):
                    decorated.__module__ = m.__module__
                    break
            else:
                decorated.__module__ = getattr(module, '__name__', '')
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


def trace_module(name):
    register_post_import_hook(set_trace, name)

#############################################################################


@decorator
def threadsafe_coroutine(fn, _, args, kwargs):
    coro = fn(*args, **kwargs)
    coro.send(None)

    class Synchronized(ObjectProxy):
        lock = RLock()

        def send(self, item):
            with self.lock:
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
