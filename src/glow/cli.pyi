from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from ._types import Get

_T = TypeVar('_T')

@dataclass
class Meta:
    help: str = ...
    flag: str | None = ...

@overload
def arg(
    default: _T,
    /,
    *,
    flag: str = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool = ...,
    help: str = ...,
    compare: bool = ...,
    metadata: Mapping[str, object] = ...,
) -> _T: ...
@overload
def arg(
    *,
    factory: Get[_T],
    flag: str = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool = ...,
    help: str = ...,
    compare: bool = ...,
    metadata: Mapping[str, object] = ...,
) -> _T: ...
@overload
def arg(
    *,
    flag: str = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool = ...,
    help: str = ...,
    compare: bool = ...,
    metadata: Mapping[str, object] = ...,
) -> Any: ...
def parse_args(
    fn: Callable[..., _T], args: Sequence[str] = ..., prog: str = ...
) -> tuple[_T, ArgumentParser]: ...
