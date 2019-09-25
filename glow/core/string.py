__all__ = ('countable', 'decimate', 'mangle', 'repr_as_obj')

from collections import Counter


_names = Counter()


def mangle():
    """
    Appends number to already seen strings, making them distinct

        >>> mangled = mangle()
        >>> mangled('a')
        'a'
        >>> mangled('b')
        'b'
        >>> mangled('a')
        'a:1'
    """
    store = Counter()

    def call(name: str):
        if name is None:
            return None

        seen = store[name]
        if seen:
            name = f'{name}:{seen}'
        store[name] += 1
        return name

    return call


def countable():
    """
    Accumulates and enumerates objects. Readable alternative to `id()`.

        >>> id_ = countable()
        >>> id_('a')
        0
        >>> id_('b')
        1
        >>> id_('a')
        0

    """
    instances = {}
    return (lambda obj: instances.setdefault(id(obj), len(instances)))


def repr_as_obj(d: dict) -> str:
    return ', '.join(f'{key}={value!r}' for key, value in d.items())


def decimate(val: int, base=1024) -> tuple:
    """Converts value to prefixed string, like `decimate(2**20) -> (1, 'M')`"""
    suffixes = 'KMGTPEZY'
    suffixes = ((i, p) for i, p in enumerate(suffixes, 1) if base ** i <= val)
    scale, suffix = max(suffixes, default=(0, ''))
    return (val / base ** scale, suffix)
