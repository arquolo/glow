import asyncio
import concurrent.futures as cf
from collections.abc import Iterable, Sequence
from typing import Protocol, Self, overload

from ._dev import drop_tb_frames
from ._locking import maybe_future
from ._more import each_is
from ._types import AUnary, Maybe, Unary

type Job[T, R] = tuple[T, cf.Future[R]]
type AJob[T, R] = tuple[T, asyncio.Future[R]]
type AnyFuture[R] = cf.Future[R] | asyncio.Future[R]

# batch -> N first items to pick, 0 if too early to yield
type UsableSize[T] = Unary[list[T], int]
type BatchFn[T, R] = Unary[list[T], Sequence[R]]
type BatchFnRv[T, R] = Unary[Iterable[T], list[R]]
type ABatchFn[T, R] = AUnary[list[T], Sequence[R]]
type ABatchFnRv[T, R] = AUnary[Iterable[T], list[R]]


class BatchDecorator(Protocol):
    @overload
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


class PsBatchDecorator[T](Protocol):
    @overload
    def __call__[R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def __call__[R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


def get_usable_size(batch_size: int, seq: Sequence) -> int:
    return batch_size if len(seq) >= batch_size else 0


def dispatch[T, R](fn: BatchFn[T, R], *xs: Job[T, R]) -> None:
    if not xs:
        return

    with _Dispatcher([f for _, f in xs], sync=True) as dsp:
        ret = fn([x for x, _ in xs])
        dsp.update(ret)


async def adispatch[T, R](fn: ABatchFn[T, R], *xs: AJob[T, R]) -> None:
    if not xs:
        return
    with _Dispatcher([f for _, f in xs], sync=False) as dsp:
        ret = await fn([x for x, _ in xs])
        dsp.update(ret)


class _Dispatcher[T]:
    def __init__(self, fs: Sequence[AnyFuture[T]], sync: bool) -> None:
        self.fs = fs
        self.sync = sync

    def __enter__(self) -> Self:
        return self

    def __exit__(self, tp, val: BaseException | None, tb) -> bool | None:
        if val is not None:
            drop_tb_frames(val, 1)
            if self.sync or not isinstance(val, asyncio.CancelledError):
                for f in self.fs:
                    f.set_exception(val)
                return True  # Suppress all but CancelledError
            for f in self.fs:
                f.cancel()
        return None

    def update(self, rs: Sequence[T]) -> None:
        rs_or_err: Maybe[T] = seqcheck(rs, len(self.fs))
        if isinstance(rs_or_err, Sequence):
            for f, x in zip(self.fs, rs_or_err, strict=True):
                f.set_result(x)
        else:
            for f in self.fs:
                f.set_exception(rs_or_err)


def seqcheck[T](obj: Sequence[T] | object, size: int) -> Maybe[T]:
    if not isinstance(obj, Sequence):
        return TypeError(f'Got {type(obj).__name__} instead of sequence')
    if len(obj) != size:
        return RuntimeError(f'Got {len(obj)} items instead of {size}')
    return list(obj)


def fs_to_results[K, R](
    fs: Iterable[tuple[K, cf.Future[R] | asyncio.Future[R]]],
) -> tuple[dict[K, R], BaseException | None]:
    results: dict[K, R] = {}
    errors = set[BaseException]()
    cancelled = acancelled = False
    for k, f in fs:
        # cf.Future - optimization to do acquire-release once
        if isinstance(f, cf.Future):
            try:
                obj = maybe_future(f)
            except cf.CancelledError:
                if not cancelled:
                    errors.add(cf.CancelledError())
                    cancelled = True
            else:
                if isinstance(obj, BaseException):
                    errors.add(obj)
                else:
                    [results[k]] = obj

        # asyncio.Future
        elif f.cancelled():
            if not acancelled:
                errors.add(asyncio.CancelledError())
                acancelled = True
        elif e := f.exception():
            errors.add(e)
        else:
            results[k] = f.result()

    return results, _format_errors(*errors)


def _format_errors(*errors: BaseException) -> BaseException | None:
    match errors:
        case []:
            return None
        case [err]:
            return err
        case errs if each_is(errs, Exception):
            return ExceptionGroup('Got multiple exceptions', errs)
        case errs:
            return BaseExceptionGroup('Got multiple exceptions', errs)
