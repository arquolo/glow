"""Make builtin iterators compatible with `len`.

Supports:
- str, bytes, range, tuple, list, set, deque iterators
- dict keys/values/items iterators
- OrderedDict keys/values/items iterators
- reversed
- map
- zip

Since Python 3.12 `itertools` doesn't support serialization,
thus `len` will never work for them.
"""

__all__ = ['apply']

import builtins
import functools
import operator
from collections import OrderedDict, deque
from collections.abc import Iterable

# --------------------------------- builtins ---------------------------------

len_hint = functools.singledispatch(builtins.len)

_iterables: list[Iterable] = [
    '',
    b'',
    range(0),
    (),
    [],
    {},
    {}.keys(),
    {}.values(),
    {}.items(),
    reversed(()),
    reversed([]),
    set(),
    frozenset(),
    deque(),
]
_transparent_types: tuple[type, ...] = tuple(
    it.__iter__().__class__ for it in _iterables
)
for _tp in _transparent_types:
    len_hint.register(_tp, operator.length_hint)

_odict_iter_tp: type = OrderedDict().__iter__().__class__


def _are_definitely_independent(iters) -> bool:
    return len({id(it) for it in iters}) == len(iters) and all(
        isinstance(it, _transparent_types) for it in iters
    )


@len_hint.register(zip)
def _len_zip(x) -> int:  # type: ignore[misc]
    _, iters = x.__reduce__()
    if not iters:
        return 0
    if len(iters) == 1:
        return len(iters[0])

    # Do not compute zip size when it's constructed from multiple iterables.
    # as there's currently no reliable way to check whether underlying
    # iterables are independent or not
    if _are_definitely_independent(iters):
        return min(map(len, iters))

    raise TypeError


@len_hint.register(map)
def _len_map(x) -> int:  # type: ignore[misc]
    _, (__fn, *iters) = x.__reduce__()
    if len(iters) == 1:
        return len(iters[0])

    # Same as for zip above
    if _are_definitely_independent(iters):
        return min(map(len, iters))

    raise TypeError


@len_hint.register(_odict_iter_tp)
def _len_odict_iter(x) -> int:
    _, [items] = x.__reduce__()
    return len(items)


def apply() -> None:
    builtins.len = len_hint
