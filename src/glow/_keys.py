__all__ = ['make_key']

from collections.abc import Hashable
from dataclasses import dataclass

_KWD_MARK = object()


@dataclass(frozen=True, slots=True)
class _HashedSeq:
    """Memorizes hash to not recompute it on cache search/update"""

    items: tuple
    hashvalue: int

    def __eq__(self, value: object) -> bool:
        return type(value) is _HashedSeq and self.items == value.items

    def __hash__(self) -> int:
        return self.hashvalue


def make_key(*args, **kwargs) -> Hashable:
    """Copied from functools._make_key, as private function"""
    if kwargs:
        args = sum(kwargs.items(), (*args, _KWD_MARK))
    if len(args) == 1 and type(args[0]) in {int, str}:
        return args[0]
    return _HashedSeq(args, hash(args))
