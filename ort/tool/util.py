import functools
import itertools
from collections import Counter
from collections.abc import Iterable

from wrapt import decorator

from . import export


@export
def unique(name, _names=Counter()):  # ! "_names" is mutable for the reason
    """Returns unique string"""
    if name is None:
        return None
    _names[name] += 1
    return f'{name}_{_names[name]}'


@export
@decorator
def once_per_instance(wrapped, instance, args, kwargs):
    instance.__dict__.setdefault('__results__', {})
    if wrapped not in instance.__results__:
        instance.__results__[wrapped] = wrapped(*args, **kwargs)
    return instance.__results__[wrapped]


@export
def as_function(wrapped=None, factory=list):
    """Make function from generator"""
    if wrapped is None:
        return functools.partial(as_function, factory=factory)

    @decorator
    def wrapper(wrapped, _, args, kwargs):
        return factory(wrapped(*args, **kwargs))

    return wrapper(wrapped)  # pylint: disable=no-value-for-parameter


@export
def as_iter(obj, times=None, base=Iterable):
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, base):
        return obj
    return itertools.repeat(obj, times=times)


@export
def chunked(iterable, size):
    """Yields chunks of at most `size` items from iterable"""
    iterator = iter(iterable)
    yield from iter(lambda: list(itertools.islice(iterator, size)),
                    [])


@export
def pretty_dict(d: dict, sep=', ') -> str:
    return sep.join(f'{key!s}={value!r}' for key, value in d.items())


@export
def pretty_bytes(x: int) -> str:
    prefixes = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi')
    for prefix in prefixes:
        if x < 1024:
            return f'{x:.2f} {prefix}B'
        x = x / 1024
    return f'{x:.2f} YiB'


@export
def iter_none(fn):
    return itertools.takewhile(
        lambda r: r is not None,
        itertools.starmap(fn, itertools.repeat(()))
    )
