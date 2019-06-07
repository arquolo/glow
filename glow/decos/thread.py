__all__ = 'threadlocal', 'shared_call'

import contextlib
import functools
from concurrent.futures import (ThreadPoolExecutor,
                                TimeoutError as _TimeoutError)
from threading import RLock, local
from weakref import WeakValueDictionary

from wrapt import decorator


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


def shared_call(fn=None, *, lock=None, timeout=.001, executor=None):
    if fn is None:
        return functools.partial(shared_call,
                                 lock=lock, timeout=timeout, executor=executor)

    lock = lock or RLock()
    futures = WeakValueDictionary()
    executor = executor or ThreadPoolExecutor()

    @decorator
    def wrapper(fn, _, args, kwargs):
        key = f'{fn}{args}{kwargs}'
        with lock:
            try:
                future = futures[key]
            except KeyError:
                futures[key] = future = executor.submit(fn, *args, **kwargs)
        while True:
            with contextlib.suppress(_TimeoutError):  # prevent deadlock
                return future.result(timeout=timeout)

    return wrapper(fn)
