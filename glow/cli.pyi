from __future__ import annotations  # until 3.10

from argparse import ArgumentParser
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, overload

_T = TypeVar('_T')


@overload
def arg(
        *,
        default: _T,
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool | None = ...,  # noqa: A002
        help: str | None = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] | None = ...) -> _T:
    ...


@overload
def arg(
        *,
        default_factory: Callable[[], _T],
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool | None = ...,  # noqa: A002
        help: str | None = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] | None = ...) -> _T:
    ...


@overload
def arg(
        *,
        init: bool = ...,
        repr: bool = ...,  # noqa: A002
        hash: bool | None = ...,  # noqa: A002
        help: str | None = ...,  # noqa: A002
        compare: bool = ...,
        metadata: Mapping[str, Any] | None = ...) -> Any:
    ...


def parse_args(cls: type[_T]) -> tuple[ArgumentParser, _T]:
    ...
