from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import AbstractContextManager
from typing import overload

def max_cpu_count(upper_bound: int = ..., mp: bool = ...) -> int: ...
def get_executor(
    max_workers: int, mp: bool
) -> AbstractContextManager[Executor]: ...
def buffered[
    T
](
    __iter: Iterable[T], /, *, latency: int = ..., mp: bool | Executor = ...
) -> Iterator[T]: ...
def starmap_n[
    R
](
    __func: Callable[..., R],
    __iter: Iterable[Iterable],
    /,
    *,
    max_workers: int | None = ...,
    prefetch: int | None = ...,
    mp: bool = ...,
    chunksize: int | None = ...,
    order: bool = ...,
) -> Iterator[R]: ...
@overload
def map_n[
    T, R
](
    __func: Callable[[T], R],
    __iter1: Iterable[T],
    /,
    *,
    max_workers: int | None = ...,
    prefetch: int | None = ...,
    mp: bool = ...,
    chunksize: int | None = ...,
    order: bool = ...,
) -> Iterator[R]: ...
@overload
def map_n[
    T1, T2, R
](
    __f: Callable[[T1, T2], R],
    __iter1: Iterable[T1],
    __iter2: Iterable[T2],
    /,
    *,
    max_workers: int | None = ...,
    prefetch: int | None = ...,
    mp: bool = ...,
    chunksize: int | None = ...,
    order: bool = ...,
) -> Iterator[R]: ...
@overload
def map_n[
    T1, T2, T3, R
](
    __f: Callable[[T1, T2, T3], R],
    __iter1: Iterable[T1],
    __iter2: Iterable[T2],
    __iter3: Iterable[T3],
    /,
    *,
    max_workers: int | None = ...,
    prefetch: int | None = ...,
    mp: bool = ...,
    chunksize: int | None = ...,
    order: bool = ...,
) -> Iterator[R]: ...
@overload
def map_n[
    R
](
    __func: Callable[..., R],
    __iter1: Iterable,
    __iter2: Iterable,
    __iter3: Iterable,
    __iter4: Iterable,
    /,
    *__iters: Iterable,
    max_workers: int | None = ...,
    prefetch: int | None = ...,
    mp: bool = ...,
    chunksize: int | None = ...,
    order: bool = ...,
) -> Iterator[R]: ...
def map_n_dict[
    T, K, R
](
    func: Callable[[T], R],
    obj: Mapping[K, T],
    /,
    *,
    max_workers: int | None = None,
    prefetch: int | None = 2,
    mp: bool = False,
    chunksize: int | None = None,
) -> dict[K, R]: ...
