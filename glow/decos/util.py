__all__ = 'as_function', 'close_at_exit'

import atexit
import functools


def as_function(gen=None, factory=list):
    """Transform generator to function"""
    if gen is None:
        return functools.partial(as_function, factory=factory)

    @functools.wraps(gen)
    def wrapper(*args, **kwargs):
        return factory(gen(*args, **kwargs))

    return wrapper


def close_at_exit(gen):
    @functools.wraps(gen)
    def wrapper(*args, **kwargs):
        it = gen(*args, **kwargs)
        atexit.register(it.close)
        return it
    return wrapper
