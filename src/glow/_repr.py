__all__ = ['countable', 'mangle', 'repr_as_obj', 'si', 'si_bin']

from collections import Counter
from collections.abc import Callable
from typing import cast

from wrapt import ObjectProxy


def mangle() -> Callable[[str], str | None]:
    """Appends number to already seen strings, making them distinct

    >>> mangled = mangle()
    >>> mangled('a')
    'a'
    >>> mangled('b')
    'b'
    >>> mangled('a')
    'a:1'
    """
    store = Counter[str]()

    def call(name: str) -> str | None:
        if name is None:
            return None

        seen = store[name]
        store[name] += 1
        if not seen:
            return name
        return f'{name}:{seen}'

    return call


def countable() -> Callable[[object], int]:
    """Accumulates and enumerates objects. Readable alternative to id().

    >>> id_ = countable()
    >>> id_('a')
    0
    >>> id_('b')
    1
    >>> id_('a')
    0
    """
    instances: dict[int, int] = {}
    return lambda obj: instances.setdefault(id(obj), len(instances))


def repr_as_obj(d: dict, /) -> str:
    """Returns pretty representation of dict.

    >>> repr_as_obj({'a': 1, 'b': 2})
    'a=1, b=2'
    """
    return ', '.join(f'{key}={value!r}' for key, value in d.items())


# ----------------------- number type with pretty repr -----------------------

_PREFIXES = 'qryzafpnum kMGTPEZYRQ'
_PREFIXES_BIN = _PREFIXES[_PREFIXES.index(' ') :].upper()


def _find_prefix(
    value: float | int, base: int, prefixes: str
) -> tuple[float, str]:
    threshold = base - 0.5
    origin = prefixes.find(' ') + 1
    value *= base**origin

    for prefix in prefixes:
        value /= base
        if -threshold < value < threshold:
            return value, prefix

    return value, prefixes[-1]


def _num_repr(value: float | int, binary: bool = False) -> tuple[float, str]:
    if value == 0:
        return 0, ('B' if binary else '')

    base, prefixes = (1024, _PREFIXES_BIN) if binary else (1000, _PREFIXES)
    value, prefix = _find_prefix(value, base, prefixes)

    prefix = prefix.strip()
    unit = (f'{prefix}iB' if prefix else 'B') if binary else prefix
    return value, unit


def _autospec(value: float) -> str:
    return '.3g' if -99.95 < value < 99.95 else '.0f'


class _Value(ObjectProxy):  # type: ignore[misc]
    __slots__ = ()
    __wrapped__: int | float
    binary: bool = False

    def __str__(self) -> str:
        value, unit = _num_repr(self.__wrapped__, self.binary)
        return f'{value:{_autospec(value)}}{unit}'

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self})'

    def __format__(self, format_spec: str) -> str:
        value, unit = _num_repr(self.__wrapped__, self.binary)
        if not format_spec:
            return f'{value:{_autospec(value)}}{unit}'
        if format_spec.endswith('s'):
            return f'{value:{_autospec(value)}}{unit}'.__format__(format_spec)
        return f'{value:{format_spec}}{unit}'

    def __reduce_ex__(self, _) -> tuple:  # Else no serialization
        return type(self), (self.__wrapped__,)


class _BinaryValue(_Value):
    binary: bool = True


def si[T: (int, float)](value: T) -> T:
    """Mix value with human-readable formatting,
    uses metric prefixes. Returns exact subtype of value.

    >>> s = si(10_000)
    >>> print(s)
    10k
    """
    return cast(T, _Value(value))


def si_bin[T: (int, float)](value: T) -> T:
    """Treats value as size in bytes, mixes it with binary prefix.
    Returns exact subtype of value.

    >>> s = si_bin(4096)
    >>> print(s, s + 5)
    4KiB 4101

    .. _Human readable bytes count
       https://programming.guide/java/formatting-byte-size-to-human-readable-format.html
    """
    return cast(T, _BinaryValue(value))
