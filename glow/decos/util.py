__all__ = 'as_function', 'once_per_instance'

import functools
from wrapt import decorator


@decorator
def once_per_instance(method, instance, args, kwargs):
    """
    Transform method so that it will be actually computed
    only once per each instance
    """
    cache = vars(instance).setdefault('__results__', {})
    try:
        return cache[method]
    except KeyError:
        cache[method] = method(*args, **kwargs)
    return cache[method]


def as_function(fn=None, factory=list):
    """Transform generator to function"""
    if fn is None:
        return functools.partial(as_function, factory=factory)

    @decorator
    def wrapper(fn, _, args, kwargs):
        return factory(fn(*args, **kwargs))

    return wrapper(fn)
