from collections.abc import Iterable, Sequence
from concurrent.futures import Future, wait
from functools import partial, update_wrapper
from threading import Lock
from time import monotonic, sleep
from typing import Never, Protocol, cast

from ._dev import hide_frame
from ._futures import (
    BatchFn,
    BatchFnRv,
    UsableSize,
    fs_to_results,
    get_usable_size,
)


class BatchDecorator(Protocol):
    def __call__[T, R](self, fn: BatchFn[T, R], /) -> BatchFnRv[T, R]: ...


def streaming_v2_sync[T, R](
    func: BatchFn[T, R] | None = None,
    /,
    *,
    batch_size: int | UsableSize[T] = 0,
    timeout: float = 0.1,
    workers: int = 1,
    pool_timeout: float = 20.0,
) -> BatchDecorator | BatchFnRv[T, R]:
    if func is None:
        deco = partial(
            streaming_v2_sync,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout,
        )
        return cast(BatchDecorator, deco)

    assert callable(func)
    assert workers >= 1
    assert timeout > 0

    from ._thread_quota import ThreadQuota  # noqa: PLC0415

    ex = ThreadQuota(workers + 1)
    lock = Lock()
    buf: list[T] = []
    futs: list[Future[Sequence[R]]] = [Future()]
    deadlines: list[float] = []
    if not callable(batch_size):
        batch_size = partial(get_usable_size, batch_size)

    def schedule_batch(n: int) -> float | None:
        fut = futs[0]
        batch, buf[:] = buf[:n], buf[n:]
        deadlines.clear()
        if batch:
            ex.submit_f(fut, func, batch)
            futs[0] = Future()
        if buf:
            deadlines[:] = [monotonic() + timeout]
            return timeout
        return None

    def sync_late_submit() -> Never:
        while True:
            with late_lk, lock:
                now = monotonic()
                if deadlines and (sleep_for := deadlines[0] - now) <= 0:
                    sleep_for = schedule_batch(len(buf)) or -1
            if sleep_for < 0:
                late_lk.acquire()
            else:
                sleep(sleep_for)

    late_lk = Lock()
    late_lk.acquire()
    ex.submit(sync_late_submit)

    def sync_submit(x: T) -> tuple[Future[Sequence[R]], int]:
        with lock:
            now = monotonic()
            if not buf:
                deadlines[:] = [now + timeout]
                late_lk.release()

            fut = futs[0]
            idx = len(buf)
            buf.append(x)

            if n := batch_size(buf):
                schedule_batch(n)
            elif deadlines and now >= deadlines[0]:
                schedule_batch(len(buf))

            return fut, idx

    def wrapper(xs: Iterable[T]) -> list[R]:
        pairs = [sync_submit(x) for x in xs]

        fs = {f for f, _ in pairs}
        try:
            dnd = wait(fs, pool_timeout, return_when='FIRST_EXCEPTION')
        finally:  # Cancel all not-yet-running tasks, we're beyond deadline
            for f in fs:
                f.cancel()
        if dnd.not_done:  # Some tasks timed out
            del dnd, fs  # ? Break reference cycle
            raise TimeoutError

        rs: dict[Future[Sequence[R]], Sequence[R]] = {}
        err = fs_to_results(zip(fs, fs), rs)
        if err is None:
            return [rs[f][i] for f, i in pairs]
        with hide_frame:
            raise err

    return update_wrapper(wrapper, func)
