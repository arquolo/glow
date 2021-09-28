from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, overload

_T = TypeVar('_T')


@overload
def arg(
        default: _T,
        /,
        *,
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool = ...,  # noqa: A002
        help: str = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] = ...) -> _T:
    ...


@overload
def arg(
        *,
        factory: Callable[[], _T],
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool = ...,  # noqa: A002
        help: str = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] = ...) -> _T:
    ...


@overload
def arg(
        *,
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool = ...,  # noqa: A002
        help: str = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] = ...) -> Any:
    ...


def parse_args(fn: Callable[..., _T],
               args: Sequence[str] = ...,
               prog: str = ...) -> tuple[_T, ArgumentParser]:
    ...
