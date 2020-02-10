__all__ = ('countable', 'mangle', 'repr_as_obj')

from typing import Callable, Counter, Dict

_names: Counter[str] = Counter()


def mangle() -> Callable[[str], str]:
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
    store = Counter[str]()

    def call(name: str) -> str:
        if name is None:
            return None

        seen = store[name]
        store[name] += 1
        if not seen:
            return name
        return f'{name}:{seen}'

    return call


def countable() -> Callable[[object], int]:
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
    instances: Dict[int, int] = {}
    return lambda obj: instances.setdefault(id(obj), len(instances))


def repr_as_obj(d: dict) -> str:
    """Returns pretty representation of dict.

    >>> repr_as_obj({'a': 1, 'b': 2})
    'a=1, b=2'
    """
    return ', '.join(f'{key}={value!r}' for key, value in d.items())
