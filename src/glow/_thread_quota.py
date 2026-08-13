"""
? At each moment each thread serve one executor.
+ Threads are reused over all executors.
+ Idle threads are stopped on _TIMEOUT.
? Most recently used threads used first (a.k.a. stack, more dropout).
"""

__all__ = ['ThreadQuota']

import sys
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from concurrent.futures._base import LOGGER
from concurrent.futures.thread import _WorkItem
from itertools import count
from queue import Empty, SimpleQueue
from threading import _register_atexit  # type: ignore[attr-defined]
from threading import Event, Lock, Thread
from weakref import WeakSet

if sys.version_info >= (3, 14):
    from concurrent.futures.thread import WorkerContext

    _worker_ctx = WorkerContext(lambda: None, ())
else:
    _worker_ctx = None

# TODO: investigate hangups when _WORKER_TIMEOUT <= .01
_WORKER_TIMEOUT = 1  # X seconds for worker to wait for next executor to serve

# ------------------------------- generics -----------------------------------


def _safe_call[**P, R](
    fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R | None:
    try:
        return fn(*args, **kwargs)
    except (Empty, IndexError, ValueError):
        return None


# ------------------------------ implementation ------------------------------

type _ExecutorPipe = SimpleQueue['ThreadQuota | None']

_shutdown = False  # set only by `_python_exit`
_shutdown_lock = Lock()  # Blocks worker creation on interpreter shutdown
_executors = WeakSet['ThreadQuota']()
_workers = WeakSet[Thread]()
_idle = deque[_ExecutorPipe]()


def _python_exit() -> None:
    global _shutdown  # noqa: PLW0603
    with _shutdown_lock:
        _shutdown = True

    for e in _executors:
        e.shutdown(cancel_futures=True)

    while q := _safe_call(_idle.pop):
        q.put(None)


_register_atexit(_python_exit)


def _worker(q: _ExecutorPipe) -> None:
    try:
        while executor := _safe_call(q.get, timeout=_WORKER_TIMEOUT):
            while True:
                with executor._shutdown_lock:
                    work_item = _safe_call(executor._work_items.popleft)
                    if work_item is None:  # Decrease worker usage for executor
                        executor._idle.append(1)
                        break

                if sys.version_info >= (3, 14):
                    work_item.run(_worker_ctx)  # Process task
                else:
                    work_item.run()
                if _shutdown:
                    executor._shutdown = True
                    return

            _idle.append(q)  # Mark worker as idle, LIFO/stack
            if _shutdown:
                return

    except BaseException:  # noqa: BLE001
        LOGGER.critical('Exception in worker', exc_info=True)
    finally:
        _safe_call(_idle.remove, q)  # Omit when '_idle' tracks weakrefs


class ThreadQuota(Executor):
    __slots__ = ('_fs', '_idle', '_shutdown', '_shutdown_lock', '_work_items')

    def __init__(self, max_workers: int) -> None:
        assert max_workers > 0
        self._work_items = deque[_WorkItem]()
        self._fs = set[Future]()
        self._idle = [1] * max_workers  # semaphore

        self._shutdown_lock = Lock()
        self._shutdown = False

        with _shutdown_lock:
            _executors.add(self)

    def submit[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        f = Future()  # type: ignore[var-annotated]
        self.submit_f(f, fn, *args, **kwargs)
        return f

    def submit_f[**P, R](
        self,
        f: Future[R],
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        with self._shutdown_lock, _shutdown_lock:
            if self._shutdown or _shutdown:
                msg = 'cannot schedule futures after shutdown'
                raise RuntimeError(msg)

            if sys.version_info >= (3, 14):
                self._work_items.append(_WorkItem(f, (fn, args, kwargs)))
            else:
                self._work_items.append(_WorkItem(f, fn, args, kwargs))
            self._fs.add(f)

            if _safe_call(self._idle.pop):  # Pool is not maximized yet
                if q := _safe_call(_idle.pop):  # Use idle worker
                    q.put(self)
                else:  # Scale to new worker
                    q = SimpleQueue[ThreadQuota | None]()
                    q.put(self)
                    w = Thread(target=_worker, args=[q])
                    w.start()
                    _workers.add(w)

        f.add_done_callback(self._forget)

    def _forget(self, f: Future) -> None:
        with self._shutdown_lock:
            self._fs.discard(f)

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

            if cancel_futures:
                while work_item := _safe_call(self._work_items.pop):
                    work_item.future.cancel()

            if not wait or not self._fs:
                return
            empty = Event()
            nleft = count(len(self._fs) - 1, -1)
            for f in self._fs:
                f.add_done_callback(
                    lambda _: None if next(nleft) else empty.set()
                )

        empty.wait()
