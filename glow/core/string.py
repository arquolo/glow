__all__ = 'counter', 'decimate', 'repr_as_obj', 'unique'

from collections import Counter


_names = Counter()


def unique(name):
    """Returns unique string"""
    if name is None:
        return None
    _names[name] += 1
    return f'{name}_{_names[name]}'


def counter():
    """
    Accumulates and enumerates objects. Readable alternative to `id()`.

        >>> count = counter()
        >>> count('a')
        0
        >>> count('b')
        1
        >>> count('a')
        0

    """
    instances = {}

    def call(obj):
        return instances.setdefault(id(obj), len(instances))

    return call


def repr_as_obj(d: dict) -> str:
    return ', '.join(f'{key}={value!r}' for key, value in d.items())


def decimate(value: int, base=1024) -> str:
    prefixes = 'KMGTPEZY'
    scale, prefix = max(((i, p) for i, p in enumerate(prefixes, 1)
                         if base ** i <= value), default=(0, ''))
    value = value / base ** scale
    return f'{value:.4g} {prefix}'
