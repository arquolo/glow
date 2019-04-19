import contextlib
import functools
from concurrent.futures import (ThreadPoolExecutor,
                                TimeoutError as _TimeoutError)
from threading import RLock, local
from weakref import WeakValueDictionary

from wrapt import decorator


def threadlocal(function, *args, _local=None, **kwargs):
    """Thread-local singleton factory, mimics `functools.partial`"""
    if args or kwargs:
        return functools.partial(threadlocal, function, *args,
                                 _local=local(), **kwargs)
    try:
        obj = _local.obj
    except AttributeError:
        obj = _local.obj = function(*args, **kwargs)
    return obj


def shared_call(wrapped=None, *,
                lock=None, timeout=.001, executor=ThreadPoolExecutor()):
    if wrapped is None:
        return functools.partial(shared_call,
                                 lock=lock, timeout=timeout, executor=executor)
    if lock is None:
        lock = RLock()
    futures = WeakValueDictionary()

    @decorator
    def wrapper(func, _, args, kwargs):
        key = f'{func}{args or ""}{kwargs or ""}'
        with lock:
            try:
                future = futures[key]
            except KeyError:
                futures[key] = future = executor.submit(func, *args, **kwargs)
        while True:
            with contextlib.suppress(_TimeoutError):  # prevent deadlock
                return future.result(timeout=timeout)

    return wrapper(wrapped)  # pylint: disable=no-value-for-parameter
