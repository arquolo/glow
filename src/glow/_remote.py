__all__ = ['remote']

import threading
import weakref
from collections.abc import Callable, Sequence
from functools import partial
from itertools import count
from multiprocessing import Process
from multiprocessing import SimpleQueue as MpSimpleQueue
from threading import Lock, Thread
from typing import Literal, cast

from ._parallel import max_cpu_count
from ._streaming import streaming
from ._types import Decorator, Maybe, maybe


def remote[**P, R](
    fn: Callable[P, R] | None = None,
    *,
    num_workers: int | None = None,
    chunk_size: int = 1,
    latency: float = 0.1,
    prefetch: int = 1,
) -> Decorator | Callable[P, R]:
    if fn is None:
        deco = partial(
            remote,
            num_workers=num_workers,
            chunk_size=chunk_size,
            latency=latency,
            prefetch=prefetch,
        )
        return cast(Decorator, deco)

    num_workers = num_workers or max_cpu_count(mp=True)
    assert chunk_size >= 1
    assert num_workers >= 1
    assert prefetch >= 0
    tmgr = _TaskManager(fn, num_workers=num_workers, prefetch=prefetch)
    return _Remote(tmgr, chunk_size=chunk_size, latency=latency)


class _Remote[**P, R]:
    def __init__(
        self,
        tmgr: '_TaskManager[P, R]',
        *,
        chunk_size: int,
        latency: float,
    ) -> None:
        # aggregates calls to batches to pass to each worker
        # `batch_submit` is called from only 1 thread (streaming.workers=1)
        self._batch_submit = streaming(
            tmgr.batch_submit, batch_size=chunk_size, timeout=latency
        )
        self._tmgr = tmgr
        self._results = tmgr.results
        self.close = weakref.finalize(self, tmgr.shutdown)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        [(lk, batch_idx, idx)] = self._batch_submit([(args, kwargs)])

        # Wait till tasks became resolved
        with lk:
            ret = self._results.pop((batch_idx, idx))
        if isinstance(ret, BaseException):
            raise ret
        return ret[0]


type _MpTQueue[*Ts] = MpSimpleQueue[tuple[*Ts] | None]


class _TaskManager[**P, R]:
    def __init__(
        self, func: Callable[P, R], num_workers: int, prefetch: int
    ) -> None:
        self.limit = threading.Semaphore(num_workers + prefetch)
        self.ids = count()
        self.locks: dict[int, Lock] = {}
        self.jobs_mpq: _MpTQueue[int, list[tuple[tuple, dict]]] = (
            MpSimpleQueue()
        )
        self.results_mpq: _MpTQueue[int, int, Maybe[Maybe[R]]] = (
            MpSimpleQueue()
        )

        self.workers: list[Process] = []
        self._func = func
        self._num_workers = num_workers

        self.results: dict[tuple[int, int], Maybe[R]] = {}
        self._run_lock = Lock()
        self._state: Literal['pending', 'running', 'closed'] = 'pending'

    def _start(self) -> None:
        # moves results from Mp Queue to per-call mapping
        Thread(target=self.populate_results, daemon=True).start()

        # picks jobs from Mp Queue, does compute and puts results to Mp Queue
        for _ in range(self._num_workers):
            p = Process(
                target=_remote_run,
                args=(self._func, self.jobs_mpq, self.results_mpq),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        self._state = 'running'

    def batch_submit(
        self, akws: Sequence[tuple[tuple, dict]]
    ) -> list[tuple[Lock, int, int]]:
        # Called from 1 thread, passes new batch of tasks and returns
        if not akws:
            return []
        self.limit.acquire()  # Protect queue from overloading

        with self._run_lock:
            if self._state == 'pending':
                self._start()
            if self._state == 'closed':
                self.limit.release()
                raise RuntimeError('cannot submit new task for closed remote')

            batch_idx = next(self.ids)
            lk = Lock()
            lk.acquire()
            self.locks[batch_idx] = lk

            try:
                # Serialize and send (see SimpleQueue impl) to worker [IPC]
                self.jobs_mpq.put((batch_idx, list(akws)))
            except BaseException:  # Serialization failed
                lk.release()
                self.limit.release()
                self.locks.pop(batch_idx)
                raise

        return [(lk, batch_idx, idx) for idx, _ in enumerate(akws)]

    def populate_results(self) -> None:
        stops = 0
        while True:
            out = self.results_mpq.get()  # From worker [IPC]
            if not out:
                stops += 1
                if stops == len(self.workers):
                    break
                continue
            batch_idx, n, rets = out
            self.limit.release()

            if isinstance(rets, BaseException):
                rets = [rets] * n
            for idx, ret in enumerate(rets):
                self.results[batch_idx, idx] = ret

            if lk := self.locks.pop(batch_idx, None):  # Notify waiters
                lk.release()

    def shutdown(self) -> None:  # thread YYY
        with self._run_lock:
            if self._state == 'closed':
                return
            self._state = 'closed'
            for _ in self.workers:
                self.jobs_mpq.put(None)
        for w in self.workers:
            w.join()


def _remote_run[R](
    func: Callable[..., R],
    jobs_mpq: _MpTQueue[int, list[tuple[tuple, dict]]],
    results_mpq: _MpTQueue[int, int, Maybe[Maybe[R]]],
) -> None:
    while ijobs := jobs_mpq.get():  # From main [IPC]
        batch_idx, jobs = ijobs
        rets = [maybe(func, *args, **kwargs) for args, kwargs in jobs]

        n = len(jobs)
        try:
            results_mpq.put((batch_idx, n, rets))  # To main [IPC]
        except BaseException as exc:  # noqa: BLE001
            # Serialization failed
            results_mpq.put((batch_idx, n, exc))  # To main [IPC]

    results_mpq.put(None)  # IPC to main
