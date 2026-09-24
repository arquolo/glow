__all__ = [
    'MulticastQueue',
    'Reusable',
    'RwLock',
    'Uid',
    'abs2',
    'aceil',
    'afloor',
    'afma',
    'amap',
    'amap_dict',
    'aminmax_norm',
    'apack',
    'around',
    'as_actor',
    'as_iter',
    'astarmap',
    'azip',
    'buffered',
    'cache_status',
    'call_once',
    'chunked',
    'circle',
    'clone_exc',
    'coalesce',
    'consumer',
    'countable',
    'cumsum',
    'declutter_tb',
    'eat',
    'get_executor',
    'groupby',
    'hide_frame',
    'ic',
    'ic_repr',
    'ichunked',
    'ilen',
    'imhash_hist',
    'imresize',
    'imresize_categorical',
    'imrotate',
    'init_loguru',
    'lock_seed',
    'mangle',
    'map_n',
    'map_n_dict',
    'max_cpu_count',
    'maximum_cumsum',
    'memoize',
    'memprof',
    'memtrack',
    'new_cache',
    'pascal',
    'register_post_import_hook',
    'repr_as_obj',
    'roundrobin',
    'si',
    'si_bin',
    'sizeof',
    'span_task',
    'starmap_n',
    'streaming',
    'summary',
    'threadlocal',
    'time_this',
    'timer',
    'trace',
    'trace_module',
    'when_imported',
    'whereami',
    'windowed',
]

from collections.abc import (
    AsyncGenerator,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
)
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
from types import CodeType, ModuleType
from typing import (
    Any,
    Literal,
    Protocol,
    Required,
    Self,
    SupportsInt,
    TypedDict,
    Unpack,
    overload,
)
from uuid import UUID

import numpy as np
import numpy.typing as npt
from loguru import FilterDict, FilterFunction, FormatFunction
from PIL.Image import Image

from ._futures import (
    ABatchFn,
    ABatchFnRv,
    BatchDecorator,
    BatchFn,
    BatchFnRv,
    PsBatchDecorator,
    UsableSize,
)
from ._types import (
    AbstractCache,
    ACallable,
    AnyIterable,
    ASCallable,
    CachePolicy,
    Decorator,
    Get,
    KeyFn,
    Pipe,
    PsDecorator,
    SupportsNext,
    SupportsSlice,
    Unary,
)

class _AmapKwargs(TypedDict, total=False):
    limit: Required[int]
    unordered: bool

class _Decorator(Decorator, Protocol):
    def drop[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...

class _PsDecorator[**P](PsDecorator[P], Protocol):
    def drop[R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...

class _BatchDecorator(BatchDecorator, Protocol):
    @overload
    def drop[T, R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def drop[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...

class _PsBatchDecorator[T](PsBatchDecorator[T], Protocol):
    @overload
    def drop[R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def drop[R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...

# ---------------------------------- _array ----------------------------------

def around(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = ...
) -> npt.NDArray[np.integer]: ...
def aceil(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = ...
) -> npt.NDArray[np.integer]: ...
def afloor(
    x: npt.NDArray[np.number], dtype: npt.DTypeLike = ...
) -> npt.NDArray[np.integer]: ...
def smallest_dtype(
    a_min: float | int, a_max: float | int | None = ...
) -> np.dtype: ...
def apack(
    a: npt.ArrayLike | npt.NDArray[np.integer],
    a_min: int | None = ...,
    a_max: int | None = ...,
) -> npt.NDArray[np.integer]: ...
def pascal(n: int) -> npt.NDArray[np.int64]: ...
def afma(
    a: npt.NDArray[np.integer | np.floating],
    scale: float = ...,
    bias: float = ...,
    dtype: npt.DTypeLike = ...,
) -> npt.NDArray[np.integer | np.floating]: ...
def abs2(c: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.floating]: ...
def aminmax_norm(a: np.ndarray) -> np.ndarray: ...

# ---------------------------------- _async ----------------------------------

class RwLock:
    def __init__(self) -> None: ...
    def read(self) -> AbstractAsyncContextManager: ...
    def write(self) -> AbstractAsyncContextManager: ...

class _AsyncContextIterator[T]:
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> T: ...
    def close(self) -> None: ...

class MulticastQueue[T]:
    def __init__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def __aiter__(self) -> AsyncGenerator[T]: ...
    async def get(self, idx: int) -> T: ...
    async def put(self, value: T) -> None: ...
    def subscribe(self) -> _AsyncContextIterator[T]: ...
    def close(self) -> None: ...

def astarmap[*Ts, R](
    func: ASCallable[*Ts, R],
    iterable: AnyIterable[tuple[*Ts]],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, R](
    func: ACallable[[T], R],
    iter1: AnyIterable[T],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, R](
    func: ACallable[[T, T2], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, T3, R](
    func: ACallable[[T, T2, T3], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[T, T2, T3, T4, R](
    func: ACallable[[T, T2, T3, T4], R],
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
@overload
def amap[R](
    func: ACallable[..., R],
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
    **kwargs: Unpack[_AmapKwargs],
) -> AsyncGenerator[R]: ...
async def amap_dict[K, T, T2](
    func: ACallable[[T], T2], obj: Mapping[K, T], /, *, limit: int
) -> dict[K, T2]: ...
@overload
def azip() -> AsyncGenerator[Any]: ...
@overload
def azip[T](iter1: AnyIterable[T], /) -> AsyncGenerator[tuple[T]]: ...  # noqa: RUF100,RUF102
@overload
def azip[T, T2](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], /
) -> AsyncGenerator[tuple[T, T2]]: ...
@overload
def azip[T, T2, T3](
    iter1: AnyIterable[T], iter2: AnyIterable[T2], iter3: AnyIterable[T3], /
) -> AsyncGenerator[tuple[T, T2, T3]]: ...
@overload
def azip[T, T2, T3, T4](
    iter1: AnyIterable[T],
    iter2: AnyIterable[T2],
    iter3: AnyIterable[T3],
    iter4: AnyIterable[T4],
    /,
) -> AsyncGenerator[tuple[T, T2, T3, T4]]: ...
@overload
def azip(
    iter1: AnyIterable,
    iter2: AnyIterable,
    iter3: AnyIterable,
    iter4: AnyIterable,
    iter5: AnyIterable,
    /,
    *iters: AnyIterable,
) -> AsyncGenerator[tuple]: ...

# ---------------------------------- _cache ----------------------------------

coalesce: Decorator

def cache_status() -> str: ...
def call_once[T](fn: Get[T], /) -> Get[T]: ...
@overload
def new_cache(*, ttl: float | None = ...) -> AbstractCache: ...
@overload
def new_cache(
    *,
    nbytes: SupportsInt,
    policy: CachePolicy | None = ...,
    ttl: float | None = ...,
) -> AbstractCache: ...
@overload
def new_cache(
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    policy: CachePolicy | None = ...,
    ttl: float | None = ...,
) -> AbstractCache: ...
@overload  # unbound or time-constrained
def memoize(
    *,
    batched: Literal[False] = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _Decorator: ...
@overload  # byte-capped
def memoize(
    *,
    nbytes: SupportsInt,
    batched: Literal[False] = ...,
    policy: CachePolicy | None = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _Decorator: ...
@overload  # count or byte-capped (optionally)
def memoize(
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[False] = ...,
    policy: CachePolicy | None = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _Decorator: ...
@overload  #  parametric, unbound or time-constrained
def memoize[**P](
    *,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> _PsDecorator[P]: ...
@overload  # parametric, byte-capped
def memoize[**P](
    *,
    nbytes: SupportsInt,
    policy: CachePolicy | None = ...,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> _PsDecorator[P]: ...
@overload  # parametric, count or byte-capped (optionally)
def memoize[**P](
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    policy: CachePolicy | None = ...,
    batched: Literal[False] = ...,
    key_fn: KeyFn[P],
    ttl: float | None = ...,
) -> _PsDecorator[P]: ...
@overload  # batched, unbound or time-constrained
def memoize(
    *,
    batched: Literal[True],
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _BatchDecorator: ...
@overload  # batched,  byte-capped
def memoize(
    *,
    nbytes: SupportsInt,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _BatchDecorator: ...
@overload  # batched, count or byte-capped (optionally)
def memoize(
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn = ...,
    ttl: float | None = ...,
) -> _BatchDecorator: ...
@overload  # batched, parametric, unbound or time-constrained
def memoize[T](
    *,
    batched: Literal[True],
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> _PsBatchDecorator[T]: ...
@overload  # batched, parametric, byte-capped
def memoize[T](
    *,
    nbytes: SupportsInt,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> _PsBatchDecorator[T]: ...
@overload  # batched, parametric, count or byte-capped (optionally)
def memoize[T](
    count: SupportsInt,
    *,
    nbytes: SupportsInt | None = ...,
    batched: Literal[True],
    policy: CachePolicy | None = ...,
    key_fn: KeyFn[T],
    ttl: float | None = ...,
) -> _PsBatchDecorator[T]: ...

# ------------------------------- _concurrency -------------------------------

def threadlocal[T, **P](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Get[T]: ...

# ---------------------------------- _coro -----------------------------------

def consumer[**P, R: SupportsNext](
    fn: Callable[P, R], /
) -> Callable[P, R]: ...
def call_with[**P, R](
    cm: AbstractContextManager,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R: ...
def threadsafe_iter[**P, Y, S, R](
    fn: Callable[P, Generator[Y, S, R]], /
) -> Callable[P, Generator[Y, S, R]]: ...
def summary() -> Generator[dict[Hashable, int], Hashable | None]: ...
def as_actor[T, R](
    fn: Unary[Iterable[T], Iterator[R]], /
) -> Generator[R, T]: ...

# ----------------------------------- _dev -----------------------------------

class _HideFrame:
    def __init__(self, nframes: int = ...) -> None: ...
    def __call__(self, nframes: int) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...

hide_frame: _HideFrame

def clone_exc[E: BaseException](exc: E) -> E: ...
def declutter_tb(e: BaseException, code: CodeType) -> None: ...
def lock_seed(seed: int) -> None: ...

# ----------------------------------- _ic ------------------------------------

@overload
def ic() -> None: ...
@overload
def ic[T](x: T, /) -> T: ...
@overload
def ic[*Ts](*args: *Ts) -> tuple[*Ts]: ...
def ic_repr(obj: object, width: int | None = None) -> str: ...

# ------------------------------- _import_hook -------------------------------

class _HookDecorator(Protocol):
    def __call__[F: Unary[ModuleType]](self, hook: F, /) -> F: ...

def register_post_import_hook(hook: Unary[ModuleType], name: str) -> None: ...
def when_imported(name: str) -> _HookDecorator: ...

# --------------------------------- _imutil ----------------------------------

type _U8 = npt.NDArray[np.uint8]
type _F32 = npt.NDArray[np.float32]
type _AnyImage = Path | str | Image | _F32 | _U8

def imhash_hist(x: _AnyImage, /, *, bins: int = ...) -> _F32 | None: ...
def imresize(
    img: np.ndarray,
    h: int,
    w: int,
    *,
    interpolation: int,
    blksize: int = ...,
) -> np.ndarray: ...
def imresize_categorical(
    img: np.ndarray,
    h: int,
    w: int,
    *,
    interpolation: int = ...,
    fill_value: int = ...,
    blksize: int = ...,
) -> np.ndarray: ...
def imrotate[T: (np.float32, np.uint8)](
    img: npt.NDArray[T],
    degrees: float,
    fit: bool | tuple[int, int] = ...,
    *,
    interpolation: int = ...,
    border: int = ...,
    blksize: int = ...,
) -> npt.NDArray[T]: ...
def circle(
    diameter: int = ..., gain: float = ...
) -> npt.NDArray[np.float32]: ...

# --------------------------------- _logging ---------------------------------

class _LoggerAddKwds(TypedDict, total=False):
    colorize: bool | None
    serialize: bool
    backtrace: bool
    diagnose: bool
    filter: str | FilterFunction | FilterDict

class _TaskSpanner:
    def __enter__(self) -> str: ...
    def __exit__(self, *args) -> bool | None: ...
    @overload
    def __call__[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](self, fn: ACallable[P, R]) -> ACallable[P, R]: ...

@overload
def init_loguru(
    level: str = ...,
    *,
    names: Iterable[str] | Mapping[str, Iterable[str]] = ...,
    fmt: str = ...,
    extra: bool = ...,
    **logger_add_kwargs: Unpack[_LoggerAddKwds],
) -> None: ...
@overload
def init_loguru(
    level: str = ...,
    *,
    names: Iterable[str] | Mapping[str, Iterable[str]] = ...,
    fmt: FormatFunction,
    **logger_add_kwargs: Unpack[_LoggerAddKwds],
) -> None: ...
@overload
def span_task() -> str | None: ...
@overload
def span_task(task_id: str, /) -> _TaskSpanner: ...
@overload
def span_task[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def span_task[**P, R](fn: ACallable[P, R], /) -> ACallable[P, R]: ...

# ---------------------------------- _more -----------------------------------

def as_iter[T](
    obj: Iterable[T] | T, /, limit: int | None = None
) -> Iterator[T]: ...
@overload
def windowed[T](it: SupportsSlice[T], size: int, /) -> Iterator[T]: ...
@overload
def windowed[T](it: Iterable[T], size: int, /) -> Iterator[tuple[T, ...]]: ...
@overload
def chunked[S](__it: SupportsSlice[S], size: int, /) -> Iterator[S]: ...
@overload
def chunked[T](__it: Iterable[T], size: int, /) -> Iterator[tuple[T, ...]]: ...
def ichunked[T](it: Iterable[T], size: int, /) -> Generator[Iterator[T]]: ...
def ilen(iterable: Iterable, /) -> int: ...
def eat(iterable: Iterable, /, *, daemon: bool = False) -> None: ...
def roundrobin[T](*iterables: Iterable[T]) -> Generator[T]: ...
@overload
def groupby[T, K: Hashable](
    iterable: Iterable[T], /, key: Unary[T, K]
) -> dict[K, list[T]]: ...
@overload
def groupby[T, K: Hashable, V](
    iterable: Iterable[T], /, key: Unary[T, K], value: Unary[T, V]
) -> dict[K, list[V]]: ...

# -------------------------------- _parallel ---------------------------------

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
def buffered[T](
    __iter: Iterable[T], /, *, latency: int = ..., mp: bool | Executor = ...
) -> Iterator[T]: ...
def starmap_n[R](
    __func: Callable[..., R],
    __iter: Iterable[Iterable],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[R]: ...
@overload
def map_n[T, R](
    __func: Callable[[T], R],
    __iter1: Iterable[T],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[R]: ...
@overload
def map_n[T, T2, R](
    __f: Callable[[T, T2], R],
    __iter1: Iterable[T],
    __iter2: Iterable[T2],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[R]: ...
@overload
def map_n[T, T2, T3, R](
    __f: Callable[[T, T2, T3], R],
    __iter1: Iterable[T],
    __iter2: Iterable[T2],
    __iter3: Iterable[T3],
    /,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[R]: ...
@overload
def map_n[R](
    __func: Callable[..., R],
    __iter1: Iterable,
    __iter2: Iterable,
    __iter3: Iterable,
    __iter4: Iterable,
    /,
    *__iters: Iterable,
    **kwargs: Unpack[_MapIterKwargs],
) -> Iterator[R]: ...
def map_n_dict[T, K, R](
    func: Callable[[T], R], obj: Mapping[K, T], /, **kwargs: Unpack[_MapKwargs]
) -> dict[K, R]: ...

# ---------------------------------- _pipes ----------------------------------

class _Pipe[In, Out](Pipe):
    def __init__(self, zero: In, push: Unary[In], pop: Get[Out]) -> None: ...
    def send(self, value: In) -> Out: ...

def cumsum() -> Pipe[int, int]: ...
def maximum_cumsum() -> Pipe[int, int]: ...

# --------------------------------- _profile ---------------------------------

def memprof(
    name_or_callback: str | Unary[float] | None = ..., /
) -> AbstractContextManager[None]: ...
def memtrack(
    callback: Unary[int, None] = ..., period: float = ...
) -> None: ...
@overload
def timer(
    name: str | None = ...,
    time: Get[int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def timer(
    callback: Unary[int] | None,
    time: Get[int] = ...,
    /,
    *,
    disable: bool = ...,
) -> AbstractContextManager[None]: ...
@overload
def time_this[**P, R](
    fn: Callable[P, R], /, *, name: str | None = ..., disable: bool = ...
) -> Callable[P, R]: ...
@overload
def time_this(*, name: str | None = ..., disable: bool = ...) -> Decorator: ...
def whereami(skip: int = ..., limit: int | None = ...) -> str: ...

# ---------------------------------- _repr -----------------------------------

def mangle() -> Unary[str, str | None]: ...
def countable() -> Unary[object, int]: ...
def repr_as_obj(d: dict, /) -> str: ...
def si[T: (int, float)](value: T) -> T: ...
def si_bin[T: (int, float)](value: T) -> T: ...

# -------------------------------- _reusable ---------------------------------

class Reusable[T]:
    def __init__(
        self, make: Get[T], delay: float, finalize: Unary[T] | None = ...
    ) -> None: ...
    def __call__(self) -> T: ...
    async def _get(self) -> T: ...

# --------------------------------- _sizeof ----------------------------------

def sizeof(obj: object, /) -> int: ...

# -------------------------------- _streaming --------------------------------

@overload
def streaming(
    *,
    batch_size: int | UsableSize = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> BatchDecorator: ...
@overload
def streaming[T](
    *,
    batch_size: UsableSize[T],
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> PsBatchDecorator[T]: ...
@overload
def streaming[T, R](
    func: BatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    workers: int = ...,
    pool_timeout: float | None = ...,
) -> BatchFnRv[T, R]: ...
@overload
def streaming[T, R](
    fn: ABatchFn[T, R],
    /,
    *,
    batch_size: int | UsableSize[T] = ...,
    timeout: float = ...,
    pool_timeout: float | None = ...,
) -> ABatchFnRv[T, R]: ...

# ----------------------------- _tracing -------------------------------------

trace: Decorator

def trace_module(name: str) -> None: ...

# ------------------------------ _uuid ---------------------------------------

class Uid(UUID):
    def __init__(self, obj: str | SupportsInt) -> None: ...
    @classmethod
    def v4(cls) -> Self: ...
