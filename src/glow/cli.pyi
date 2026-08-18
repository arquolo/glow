from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from ._types import Get

@dataclass
class Meta:
    help: str = ...
    flag: str | None = ...

@overload
def arg[T](
    default: T,
    /,
    *,
    flag: str = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool = ...,
    help: str = ...,
    compare: bool = ...,
    metadata: Mapping[str, object] = ...,
) -> T: ...
@overload
def arg[T](
    *,
    factory: Get[T],
    flag: str = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool = ...,
    help: str = ...,
    compare: bool = ...,
    metadata: Mapping[str, object] = ...,
) -> T: ...
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
@overload
def run[T](
    fn: Callable[..., T],
    /,
    args: Sequence[str] | None = ...,
    prog: str | None = ...,
    *,
    return_parser: Literal[False] = ...,
) -> T: ...
@overload
def run[T](
    fn: Callable[..., T],
    /,
    args: Sequence[str] | None = ...,
    prog: str | None = ...,
    *,
    return_parser: Literal[True],
) -> tuple[T, ArgumentParser]: ...
def parse_args[T](
    fn: Callable[..., T], args: Sequence[str] = ..., prog: str = ...
) -> tuple[T, ArgumentParser]: ...
