"""
? At each moment each thread serve one executor.
+ Threads are reused over all executors.
+ Idle threads are stopped on _TIMEOUT.
? Most recently used threads used first (a.k.a. stack, more dropout).
"""

__all__ = ['ThreadQuota']

import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from concurrent.futures._base import LOGGER
from concurrent.futures.thread import _WorkItem
from queue import Empty, SimpleQueue
from threading import _register_atexit  # type: ignore[attr-defined]
from threading import Lock, Thread
from weakref import WeakSet

# TODO: investigate hangups when _TIMEOUT <= .01
_TIMEOUT = 1
_MIN_IDLE = os.cpu_count() or 1

# ------------------------------- generics -----------------------------------


def _safe_call[
    **P, T
](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T | None:
    try:
        return fn(*args, **kwargs)
    except (Empty, IndexError, ValueError):
        return None


# ------------------------------ implementation ------------------------------

type _Pipe = SimpleQueue['ThreadQuota | None']

_shutdown = False  # set only by `_python_exit`
_shutdown_lock = Lock()  # Blocks worker creation on interpreter shutdown
_executors = WeakSet['ThreadQuota']()
_workers = WeakSet[Thread]()
_idle = deque[_Pipe]()


def _python_exit() -> None:
    global _shutdown  # noqa: PLW0603
    with _shutdown_lock:
        _shutdown = True

    for e in _executors:
        e.shutdown(cancel_futures=True)

    while q := _safe_call(_idle.pop):
        q.put(None)


_register_atexit(_python_exit)


def _worker(q: _Pipe) -> None:
    try:
        while executor := _safe_call(q.get, timeout=_TIMEOUT):
            while work_item := _safe_call(executor._work_queue.popleft):
                work_item.run()  # Process task
                if _shutdown:
                    executor._shutdown = True
                    return

            executor._idle.append(1)  # Decrease worker usage for executor
            _idle.append(q)  # Mark worker as idle, LIFO/stack
            if _shutdown:
                return

    except BaseException:  # noqa: BLE001
        LOGGER.critical('Exception in worker', exc_info=True)
    finally:
        if _TIMEOUT:
            _safe_call(_idle.remove, q)  # Omit when '_idle' tracks weakrefs


class ThreadQuota(Executor):
    __slots__ = ('_work_queue', '_idle', '_shutdown_lock', '_shutdown')

    def __init__(self, max_workers: int) -> None:
        self._work_queue = deque[_WorkItem]()
        self._idle = [1] * max_workers  # semaphore

        self._shutdown_lock = Lock()
        self._shutdown = False

        with _shutdown_lock:
            _executors.add(self)

    def submit[
        **P, R
    ](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        f = Future()  # type: ignore[var-annotated]
        self.submit_f(f, fn, *args, **kwargs)
        return f

    def submit_f[
        **P, R
    ](
        self,
        f: Future[R],
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        with self._shutdown_lock, _shutdown_lock:
            if self._shutdown or _shutdown:
                raise RuntimeError('cannot schedule futures after shutdown')

            self._work_queue.append(_WorkItem(f, fn, args, kwargs))

            if _safe_call(self._idle.pop):  # Pool is not maximized yet
                if q := _safe_call(_idle.pop):  # Use idle worker
                    q.put(self)
                else:  # Scale to new worker
                    q = SimpleQueue[ThreadQuota | None]()
                    q.put(self)
                    w = Thread(target=_worker, args=[q])
                    w.start()
                    _workers.add(w)

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

            if cancel_futures:
                while work_item := _safe_call(self._work_queue.pop):
                    work_item.future.cancel()

            # TODO: if not `wait` - stop sub-workers
            if not _TIMEOUT:
                # Keep at most 25% of workers idle
                while len(_idle) > max(len(_workers) / 4, _MIN_IDLE) and (
                    q := _safe_call(_idle.popleft)
                ):
                    q.put(None)
