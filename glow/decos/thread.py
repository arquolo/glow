__all__ = ('call_once', 'threadlocal', 'interpreter_lock', 'shared_call')

import sys
import functools
from concurrent.futures import Future
from contextlib import contextmanager, ExitStack
from threading import RLock, local
from weakref import WeakValueDictionary


def threadlocal(fn, *args, _local=None, **kwargs):
    """Thread-local singleton factory, mimics `functools.partial`"""
    if args or kwargs:
        return functools.partial(
            threadlocal, fn, *args, _local=local(), **kwargs
        )
    try:
        return _local.obj
    except AttributeError:
        _local.obj = fn(*args, **kwargs)
        return _local.obj


@contextmanager
def interpreter_lock(timeout=1_000):
    """
    Completely forbids thread switching in underlying scope.
    Thus makes it fully thread-safe, although adds high performance penalty.

    >>> import threading, time
    >>> from concurrent.futures import ThreadPoolExecutor
    >>> value = 0
    >>> steps = 1_000_000
    >>> def writer(_):
    ...     nonlocal value
    ...     with interpreter_lock():  # increment by 2
    ...         value += 1
    ...         # <- won't switch here ->
    ...         value += 1
    ...
    >>> def reader(_):
    ...     time.sleep(0)
    ...     return value % 2  # should be 0
    ...
    >>> with ThreadPoolExecutor() as pool:
    ...     pool.map(writer, range(steps))
    ...     sum(f.result() for f in pool.map(reader, range(steps)))
    ...
    0
    """
    default = sys.getswitchinterval()
    sys.setswitchinterval(timeout)
    try:
        yield
    finally:
        sys.setswitchinterval(default)


class Stack(ExitStack):
    def defer(self, fn, *args, **kwargs):
        def apply(future):
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        future = Future()
        self.callback(apply, future)
        return future


def call_once(fn):
    """Makes `fn()` callable a singleton"""
    lock = RLock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with Stack() as stack:
            with lock:
                if fn.__future__ is None:
                    fn.__future__ = stack.defer(fn, *args, **kwargs)

        return fn.__future__.result()

    fn.__future__ = None
    return wrapper


def shared_call(fn=None, *, lock=None):
    """Merges concurrent calls to `fn` with the same `args` to single one"""
    if fn is None:
        return functools.partial(shared_call, lock=lock)

    lock = lock or RLock()
    futures = WeakValueDictionary()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'

        with Stack() as stack:
            with lock:
                try:
                    future = futures[key]
                except KeyError:
                    futures[key] = future = stack.defer(fn, *args, **kwargs)

        return future.result()

    return wrapper
