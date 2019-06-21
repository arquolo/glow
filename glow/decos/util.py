__all__ = 'as_function', 'call_once'

import functools
from concurrent.futures import ThreadPoolExecutor
from threading import RLock


def call_once(fn):
    """
    Transform `fn` so that it will be actually called only at first call
    """
    pool = ThreadPoolExecutor(1)
    lock = RLock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with lock:
            if fn.__future__ is None:
                fn.__future__ = pool.submit(fn, *args, **kwargs)
        return fn.__future__.result()

    fn.__future__ = None
    return wrapper


def as_function(gen=None, factory=list):
    """Transform generator to function"""
    if gen is None:
        return functools.partial(as_function, factory=factory)

    @functools.wraps(gen)
    def wrapper(*args, **kwargs):
        return factory(gen(*args, **kwargs))

    return wrapper
