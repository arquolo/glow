__all__ = ['env']

import sys
from collections.abc import Generator, Iterator, KeysView, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from . import repr_as_obj

if sys.version_info >= (3, 15):
    _empty = frozendict()  # noqa: F821
else:
    from types import MappingProxyType

    _empty = MappingProxyType({})


class _Env:
    """Environment with scopes.

    The outermost scope takes highest priority.
    As example:

        >>> print(env)  # Empty in global scope
        _Env()
        >>> with env(a=1):  # Overrides `a` from inner scope
        ...     print(env)
        ...     with env(a=2, b=3):  # Set defaults for `a` and `b`
        ...         print(env)
        ...     print(env)
        _Env(a=1)
        _Env(a=1, b=3)
        _Env(a=1)
        >>> print(env)  # Reset to initial state
        _Env()

    """

    __slots__ = ('_cv',)

    def __init__(self) -> None:
        self._cv = ContextVar[Mapping[str, Any]]('env', default=_empty)

    # `list(x)` compat

    def __iter__(self) -> Iterator[str]:
        return iter(self._cv.get())

    # `dict(env)` and `{**env}` compat

    def __getitem__(self, key: str, /) -> Any:
        return self._cv.get()[key]

    def keys(self) -> KeysView[str]:
        return self._cv.get().keys()

    # `key in env` compat

    def __contains__(self, key: object, /) -> bool:
        return key in self._cv.get()

    # `env == {}` compat

    def __eq__(self, rhs: object, /) -> bool:
        return self._cv.get() == rhs

    __hash__ = None  # type: ignore[assignment]

    # core logic

    if sys.version_info >= (3, 14):

        @contextmanager
        def __call__(self, **items) -> Generator[None]:
            with self._cv.set({**items, **self._cv.get()}):
                yield
    else:

        @contextmanager
        def __call__(self, **items) -> Generator[None]:
            token = self._cv.set({**items, **self._cv.get()})
            try:
                yield
            finally:
                self._cv.reset(token)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr_as_obj({**self._cv.get()})})'


env = _Env()
