__all__ = ('countable', 'mangle', 'repr_as_obj')

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
    """Returns pretty representation of dict.

    >>> repr_as_obj({'a': 1, 'b': 2})
    'a=1, b=2'
    """
    return ', '.join(f'{key}={value!r}' for key, value in d.items())
