from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import AbstractContextManager
from typing import TypedDict, TypeVar, Unpack, overload

_T = TypeVar('_T')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')
_K = TypeVar('_K')
_R = TypeVar('_R')

class _MapKwargs(TypedDict, total=False):
    max_workers: int | None
    prefetch: int | None
    mp: bool
    chunksize: int | None

class _MapIterKwargs(_MapKwargs, total=False):
    unordered: bool

def max_cpu_count(upper_bound: int = ..., *, mp: bool = ...) -> int: ...
def get_executor(
    max_workers: int, mp: bool
) -> AbstractContextManager[Executor]: ...
def buffered(
    __iter: Iterable[_T], /, *, latency: int = ..., mp: bool | Executor = ...
) -> Iterator[_T]: ...
def starmap_n(
    __func: Callable[..., _R],
    __iter: Iterable[Iterable],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[_R]: ...
@overload
def map_n(
    __func: Callable[[_T], _R],
    __iter1: Iterable[_T],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[_R]: ...
@overload
def map_n(
    __f: Callable[[_T, _T2], _R],
    __iter1: Iterable[_T],
    __iter2: Iterable[_T2],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[_R]: ...
@overload
def map_n(
    __f: Callable[[_T, _T2, _T3], _R],
    __iter1: Iterable[_T],
    __iter2: Iterable[_T2],
    __iter3: Iterable[_T3],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[_R]: ...
@overload
def map_n(
    __func: Callable[..., _R],
    __iter1: Iterable,
    __iter2: Iterable,
    __iter3: Iterable,
    __iter4: Iterable,
    /,
    *__iters: Iterable,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[_R]: ...
def map_n_dict(
    func: Callable[[_T], _R],
    obj: Mapping[_K, _T],
    /,
    **kwargs: Unpack[_MapKwargs],
) -> dict[_K, _R]: ...
