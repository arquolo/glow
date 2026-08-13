import asyncio
import concurrent.futures as cf
from collections.abc import Iterable, Sequence
from typing import Protocol, overload

from ._dev import hide_frame
from ._more import each_is
from ._types import AUnary, Unary

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
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...


class PsBatchDecorator[T](Protocol):
    def __call__[R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...


class ABatchDecorator(Protocol):
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


class PsABatchDecorator[T](Protocol):
    def __call__[R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


class AnyBatchDecorator(Protocol):
    @overload
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


class PsAnyBatchDecorator[T](Protocol):
    @overload
    def __call__[R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...
    @overload
    def __call__[R](self, fn: ABatchFn[T, R], /) -> ABatchFnRv[T, R]: ...


def get_usable_size(batch_size: int, seq: Sequence) -> int:
    return batch_size if len(seq) >= batch_size else 0


def dispatch[T, R](fn: BatchFn[T, R], *xs: Job[T, R]) -> None:
    if not xs:
        return

    try:
        with hide_frame:
            ret = fn([x for x, _ in xs])
    except BaseException as exc:  # noqa: BLE001
        for _, f in xs:
            f.set_exception(exc)
    else:
        _populate_futures(ret, [f for _, f in xs])


async def adispatch[T, R](fn: ABatchFn[T, R], *xs: AJob[T, R]) -> None:
    if not xs:
        return

    try:
        with hide_frame:
            ret = await fn([x for x, _ in xs])
    except asyncio.CancelledError:
        for _, f in xs:
            f.cancel()
        raise
    except BaseException as exc:  # noqa: BLE001
        for _, f in xs:
            f.set_exception(exc)
    else:
        _populate_futures(ret, [f for _, f in xs])


def _populate_futures[T](ret, fs: Sequence[AnyFuture[T]]) -> None:
    err: Exception
    if isinstance(ret, Sequence):
        if (nf := len(fs)) == (n := len(ret)):
            for f, x in zip(fs, ret):
                f.set_result(x)
            return
        err = RuntimeError(f'Call with {nf} arguments returned {n} results')
    else:
        err = TypeError(f'Returned {type(ret).__name__} instead of sequence')
    for f in fs:
        f.set_exception(err)


def fs_to_results[K, F: AnyFuture, R](
    fs: Iterable[tuple[K, F]], results: dict[K, R]
) -> BaseException | None:
    errors = set[BaseException]()
    sync_cancelled = async_cancelled = False
    for k, f in fs:
        if f.cancelled():
            match f:
                case cf.Future() if not sync_cancelled:
                    errors.add(cf.CancelledError())
                    sync_cancelled = True
                case asyncio.Future() if not async_cancelled:
                    errors.add(asyncio.CancelledError())
                    async_cancelled = True
        elif e := f.exception():
            errors.add(e)
        else:
            results[k] = f.result()

    match list(errors):
        case []:
            return None
        case [err]:
            return err
        case errs if each_is(errs, Exception):
            return ExceptionGroup('Got multiple exceptions', errs)
        case errs:
            return BaseExceptionGroup('Got multiple exceptions', errs)
