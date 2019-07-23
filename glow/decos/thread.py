__all__ = 'call_once', 'threadlocal', 'shared_call'

import functools
from concurrent.futures import Future
from contextlib import ExitStack
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


def _defer(fn, future, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        future.set_exception(exc)
    else:
        future.set_result(result)


def call_once(fn):
    """Makes `fn()` callable a singleton"""
    lock = RLock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with ExitStack() as stack:
            with lock:
                if fn.__future__ is None:
                    fn.__future__ = Future()
                    stack.callback(_defer, fn, fn.__future__, *args, **kwargs)

        return fn.__future__.result()

    fn.__future__ = None
    return wrapper


def shared_call(fn=None, *, lock=None):
    """
    Makes `fn()` callable a timed singleton.
    Allows reuse of result of `fn` among callers.
    """
    if fn is None:
        return functools.partial(shared_call, lock=lock)

    lock = lock or RLock()
    futures = WeakValueDictionary()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'

        with ExitStack() as stack:
            with lock:
                try:
                    future = futures[key]
                except KeyError:
                    futures[key] = future = Future()
                    stack.callback(_defer, fn, future, *args, **kwargs)

        return future.result()

    return wrapper
