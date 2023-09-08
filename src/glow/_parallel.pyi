from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import AbstractContextManager
from typing import Generic, Protocol, TypeVar, overload

_T = TypeVar('_T')
_T1_contra = TypeVar('_T1_contra', contravariant=True)
_T2_contra = TypeVar('_T2_contra', contravariant=True)
_T3_contra = TypeVar('_T3_contra', contravariant=True)
_K_co = TypeVar('_K_co', covariant=True)
_R_co = TypeVar('_R_co', covariant=True)


class _Callable1(Generic[_T1_contra, _R_co], Protocol):
    def __call__(self, __1: _T1_contra, /) -> _R_co:
        ...


class _Callable2(Generic[_T1_contra, _T2_contra, _R_co], Protocol):
    def __call__(self, __1: _T1_contra, __2: _T2_contra, /) -> _R_co:
        ...


class _Callable3(Generic[_T1_contra, _T2_contra, _T3_contra, _R_co], Protocol):
    def __call__(self, __1: _T1_contra, __2: _T2_contra, __3: _T3_contra,
                 /) -> _R_co:
        ...


class _Callable4(Generic[_R_co], Protocol):
    def __call__(self, __1, __2, __3, __4, *args) -> _R_co:
        ...


def max_cpu_count(upper_bound: int = ..., mp: bool = ...) -> int:
    ...


def get_executor(max_workers: int,
                 mp: bool) -> AbstractContextManager[Executor]:
    ...


def buffered(__iter: Iterable[_T],
             /,
             *,
             latency: int = ...,
             mp: bool | Executor = ...) -> Iterator[_T]:
    ...


def starmap_n(__func: Callable[..., _R_co],
              __iter: Iterable[Iterable],
              /,
              *,
              max_workers: int | None = ...,
              prefetch: int | None = ...,
              mp: bool = ...,
              chunksize: int | None = ...,
              order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__func: _Callable1[_T1_contra, _R_co],
          __iter1: Iterable[_T1_contra],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__f: _Callable2[_T1_contra, _T2_contra, _R_co],
          __iter1: Iterable[_T1_contra],
          __iter2: Iterable[_T2_contra],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__f: _Callable3[_T1_contra, _T2_contra, _T3_contra, _R_co],
          __iter1: Iterable[_T1_contra],
          __iter2: Iterable[_T2_contra],
          __iter3: Iterable[_T3_contra],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__func: _Callable4[_R_co],
          /,
          __iter1: Iterable,
          __iter2: Iterable,
          __iter3: Iterable,
          __iter4: Iterable,
          *__iters: Iterable,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


def map_n_dict(func: Callable[[_T1_contra], _R_co],
               obj: Mapping[_K_co, _T1_contra],
               /,
               *,
               max_workers: int | None = None,
               prefetch: int | None = 2,
               mp: bool = False,
               chunksize: int | None = None) -> dict[_K_co, _R_co]:
    ...
