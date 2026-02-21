import asyncio
import concurrent.futures as cf
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Protocol, overload

from ._dev import hide_frame
from ._types import Coro, Some

type AnyFuture[R] = cf.Future[R] | asyncio.Future[R]
type Job[T, R] = tuple[T, AnyFuture[R]]

type BatchFn[T, R] = Callable[[Sequence[T]], Sequence[R]]
type ABatchFn[T, R] = Callable[[Sequence[T]], Coro[Sequence[R]]]


class BatchDecorator(Protocol):
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
class ABatchDecorator(Protocol):
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...


class AnyBatchDecorator(Protocol):
    @overload
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
    @overload
    def __call__[T, R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...
class PsAnyBatchDecorator[T](Protocol):
    @overload
    def __call__[R](self, fn: BatchFn[T, R], /) -> BatchFn[T, R]: ...
    @overload
    def __call__[R](self, fn: ABatchFn[T, R], /) -> ABatchFn[T, R]: ...


def dispatch[T, R](fn: BatchFn[T, R], *xs: Job[T, R]) -> None:
    if not xs:
        return

    obj: Some[Sequence[R]] | BaseException
    try:
        with hide_frame:
            ret = fn([x for x, _ in xs])
    except BaseException as exc:  # noqa: BLE001
        obj = exc
    else:
        obj = _check_protocol(ret, len(xs))

    if isinstance(obj, Some):
        for (_, f), res in zip(xs, obj.x):
            f.set_result(res)
    else:
        for _, f in xs:
            f.set_exception(obj)


async def adispatch[T, R](fn: ABatchFn[T, R], *xs: Job[T, R]) -> None:
    if not xs:
        return

    obj: Some[Sequence[R]] | BaseException
    try:
        with hide_frame:
            ret = await fn([x for x, _ in xs])
    except asyncio.CancelledError:
        for _, f in xs:
            f.cancel()
        raise
    except BaseException as exc:  # noqa: BLE001
        obj = exc
    else:
        obj = _check_protocol(ret, len(xs))

    if isinstance(obj, Some):
        for (_, f), res in zip(xs, obj.x):
            f.set_result(res)
    else:
        for _, f in xs:
            f.set_exception(obj)
            if isinstance(f, asyncio.Future):
                f.exception()  # Mark exception as retrieved


def _check_protocol[S: Sequence](ret: S, n: int) -> Some[S] | BaseException:
    if not isinstance(ret, Sequence):
        return TypeError(
            f'Call returned non-sequence. Got {type(ret).__name__}'
        )
    if len(ret) != n:
        return RuntimeError(
            f'Call with {n} arguments '
            f'incorrectly returned {len(ret)} results'
        )
    return Some(ret)


def gather_fs[K: Hashable, R](
    fs: Iterable[tuple[K, AnyFuture[R]]],
) -> tuple[dict[K, R], BaseException | None]:
    results: dict[K, R] = {}
    errors = set[BaseException]()
    default: BaseException | None = None
    for k, f in fs:
        if f.cancelled():
            exc_tp = _fut_tp_to_cancel_tp.get(type(f))
            assert exc_tp, f'Unknown future type: {type(f).__qualname__}'
            assert default is None or isinstance(default, exc_tp)
            default = exc_tp()
        elif e := f.exception():
            errors.add(e)
        else:
            results[k] = f.result()

    match list(errors):
        case []:
            return (results, default)
        case [err]:
            return (results, err)
        case errs:
            msg = 'Got multiple exceptions'
            if all(isinstance(e, Exception) for e in errs):
                err = ExceptionGroup(msg, errs)  # type: ignore[type-var]
            else:
                err = BaseExceptionGroup(msg, errs)
            return (results, err)


_fut_tp_to_cancel_tp: dict[type[AnyFuture], type[BaseException]] = {
    cf.Future: cf.CancelledError,
    asyncio.Future: asyncio.CancelledError,
}
