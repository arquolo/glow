__all__ = ('as_function', )

import functools


def as_function(gen=None, factory=list):
    """Transform generator to function"""
    if gen is None:
        return functools.partial(as_function, factory=factory)

    @functools.wraps(gen)
    def wrapper(*args, **kwargs):
        return factory(gen(*args, **kwargs))

    return wrapper
