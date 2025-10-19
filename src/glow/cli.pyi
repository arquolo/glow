from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

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
def parse_args[T](
    fn: Callable[..., T], args: Sequence[str] = ..., prog: str = ...
) -> tuple[T, ArgumentParser]: ...
