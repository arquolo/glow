__all__ = 'threadlocal', 'shared_call'

import contextlib
import functools
from concurrent.futures import (Future, TimeoutError as _TimeoutError)
from threading import RLock, local
from weakref import WeakValueDictionary


def as_future(fn, *args, **kwargs):
    fut = Future()
    try:
        result = fn(*args, **kwargs)
    except BaseException as exception:
        fut.set_exception(exception)
    else:
        fut.set_result(result)
    return fut


def threadlocal(fn, *args, _local=None, **kwargs):
    """Thread-local singleton factory, mimics `functools.partial`"""
    if args or kwargs:
        return functools.partial(threadlocal, fn, *args,
                                 _local=local(), **kwargs)
    try:
        return _local.obj
    except AttributeError:
        _local.obj = fn(*args, **kwargs)
        return _local.obj


def shared_call(fn=None, *, lock=None, timeout=.001):
    if fn is None:
        return functools.partial(shared_call, lock=lock, timeout=timeout)

    lock = lock or RLock()
    futures = WeakValueDictionary()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'
        with lock:
            try:
                future = futures[key]
            except KeyError:
                futures[key] = future = as_future(fn, *args, **kwargs)
        while True:
            with contextlib.suppress(_TimeoutError):  # prevent deadlock
                return future.result(timeout=timeout)

    return wrapper
