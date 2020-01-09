__all__ = ('trace', 'trace_module', 'summary')

import contextlib
import inspect
import threading
import types
from typing import Counter, Generator, TypeVar

import wrapt

_T = TypeVar('_T')


def _break_on_globe(frame_infos):
    for info in frame_infos:
        yield info
        if info.function == '<module>':
            return


def whereami(skip=2):
    frame_infos = _break_on_globe(inspect.stack()[skip:])
    return ' -> '.join(
        ':'.join((
            getattr(inspect.getmodule(info.frame), '__name__', '[root]'),
            info.function,
            str(info.lineno),
        ))
        for info in list(frame_infos)[::-1]
    )


@wrapt.decorator
def trace(fn, _, args, kwargs):
    print(f'<({whereami(3)})> : {fn.__module__ or ""}.{fn.__qualname__}',
          flush=True)
    return fn(*args, **kwargs)

# TODO: rewrite using unittest.mock

def set_trace(obj, seen=None, prefix=None, module=None):
    if isinstance(obj, types.ModuleType):
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
        with contextlib.suppress(AttributeError, TypeError):
            member = getattr(obj, name)
            if not callable(member):
                continue
            decorated = trace(member)

            for m in (decorated, member, obj):
                with contextlib.suppress(AttributeError):
                    decorated.__module__ = m.__module__
                    break
            else:
                decorated.__module__ = getattr(module, '__name__', '')
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


def trace_module(name):
    wrapt.register_post_import_hook(set_trace, name)

#############################################################################


@wrapt.decorator
def threadsafe_coroutine(fn, _, args, kwargs):
    coro = fn(*args, **kwargs)
    coro.send(None)

    class Synchronized(wrapt.ObjectProxy):
        lock = threading.RLock()

        def send(self, item):
            with self.lock:
                return self.__wrapped__.send(item)

        def __next__(self):
            return self.send(None)

    return Synchronized(coro)


@threadsafe_coroutine
def summary() -> Generator[None, _T, None]:
    state = Counter[_T]()
    while True:
        key = yield
        if key is None:
            state.clear()
            continue
        state[key] += 1
        print(dict(state), flush=True, end='\r')
