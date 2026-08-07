import asyncio
import concurrent.futures as cf
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, overload

from ._dev import hide_frame
from ._types import Coro, Maybe, Some

type Job[T, R] = tuple[T, cf.Future[R]]
type AJob[T, R] = tuple[T, asyncio.Future[R]]
type AnyFuture[R] = cf.Future[R] | asyncio.Future[R]

# batch -> N first items to pick, 0 if too early to yield
type UsableSize[T] = Callable[[list[T]], int]
type BatchFn[T, R] = Callable[[list[T]], Sequence[R]]
type BatchFnRv[T, R] = Callable[[Iterable[T]], list[R]]
type ABatchFn[T, R] = Callable[[list[T]], Coro[Sequence[R]]]
type ABatchFnRv[T, R] = Callable[[Iterable[T]], Coro[list[R]]]


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

    obj: Maybe[list[R]]
    try:
        with hide_frame:
            ret = list(fn([x for x, _ in xs]))
    except BaseException as exc:  # noqa: BLE001
        obj = exc
    else:
        obj = _maybe_make_some(ret, len(xs))

    if isinstance(obj, Some):
        for (_, f), res in zip(xs, obj.x):
            f.set_result(res)
    else:
        for _, f in xs:
            f.set_exception(obj)


async def adispatch[T, R](fn: ABatchFn[T, R], *xs: AJob[T, R]) -> None:
    if not xs:
        return

    obj: Maybe[list[R]]
    try:
        with hide_frame:
            ret = list(await fn([x for x, _ in xs]))
    except asyncio.CancelledError:
        for _, f in xs:
            f.cancel()
        raise
    except BaseException as exc:  # noqa: BLE001
        obj = exc
    else:
        obj = _maybe_make_some(ret, len(xs))

    if isinstance(obj, Some):
        for (_, f), res in zip(xs, obj.x):
            f.set_result(res)
    else:
        for _, f in xs:
            f.set_exception(obj)
            if isinstance(f, asyncio.Future):
                f.exception()  # Mark exception as retrieved


def _maybe_make_some[S: Sequence](ret: S, n: int) -> Maybe[S]:
    if not isinstance(ret, Sequence):
        return TypeError(
            f'Call returned non-sequence. Got {type(ret).__name__}'
        )
    if len(ret) != n:
        return RuntimeError(
            f'Call with {n} arguments incorrectly returned {len(ret)} results'
        )
    return Some(ret)


def gather_fs[K, R](
    fs: Mapping[K, AnyFuture[R]],
) -> tuple[dict[K, R], BaseException | None]:
    if not fs:
        return {}, None

    results: dict[K, R] = {}
    errors = set[BaseException]()
    sync_cancelled = async_cancelled = False
    for k, f in fs.items():
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
            return (results, None)
        case [err]:
            return (results, err)
        case errs:
            msg = 'Got multiple exceptions'
            if all(isinstance(e, Exception) for e in errs):
                err = ExceptionGroup(msg, errs)  # type: ignore[type-var]
            else:
                err = BaseExceptionGroup(msg, errs)
            return (results, err)
