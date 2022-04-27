from __future__ import annotations

__all__ = ['stream_batched', 'call_once', 'threadlocal', 'shared_call']

import threading
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Any, NoReturn, TypeVar, cast
from weakref import WeakValueDictionary

from .util import make_key

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_Make = Callable[[], _T]
_ZeroArgsF = TypeVar('_ZeroArgsF', bound=Callable[[], Any])

_unset = object()


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    """Thread-local singleton factory, mimics `functools.partial`"""
    local_ = threading.local()

    def wrapper() -> _T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)


@dataclass
class _UniqueTask:
    _fn: Callable[[], Any]
    _lock: Lock = field(default_factory=Lock)
    _result: Any = _unset
    _exception: BaseException | None = None

    def run(self):
        with self._lock:
            if self._exception:
                raise self._exception
            if self._result is not _unset:
                return self._result

            try:
                r = self._fn()
            except BaseException as e:
                self._exception = e
                raise
            else:
                self._result = r
                return r


def call_once(fn: _ZeroArgsF) -> _ZeroArgsF:
    """Makes `fn()` callable a singleton.
    DO NOT USE with recursive functions"""
    def wrapper():
        return task.run()

    fn._task = task = _UniqueTask(fn)  # type: ignore
    return cast(_ZeroArgsF, update_wrapper(wrapper, fn))


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one.
    DO NOT USE with recursive functions"""
    tasks = WeakValueDictionary[Hashable, _UniqueTask]()
    lock = Lock()

    def wrapper(*args, **kwargs):
        key = make_key(*args, **kwargs)

        with lock:  # Create only one task per args-kwargs set
            if not (task := tasks.get(key)):
                tasks[key] = task = _UniqueTask(partial(fn, *args, **kwargs))

        return task.run()

    return cast(_F, update_wrapper(wrapper, fn))


def _batch_invoke(func: _Make[Sequence[_T]], futures: Sequence[Future[_T]]):
    try:
        results = func()
        assert len(results) == len(futures)
    except BaseException as exc:  # noqa: PIE786
        for fut in futures:
            fut.set_exception(exc)
    else:
        for fut, res in zip(futures, results):
            fut.set_result(res)


def stream_batched(func=None, *, batch_size, latency=0.1, timeout=20.):
    """
    Delays start of computation up to `latency` seconds
    in order to fill batch to batch_size items and
    send it at once to target function.
    `timeout` specifies timeout to wait results from worker.

    Simplified version of https://github.com/ShannonAI/service-streamer
    """
    if func is None:
        return partial(
            stream_batched,
            batch_size=batch_size,
            latency=latency,
            timeout=timeout)

    assert callable(func)
    buf = SimpleQueue()

    def _fetch_batch():
        end_time = monotonic() + latency
        for _ in range(batch_size):
            try:
                yield buf.get(timeout=end_time - monotonic())
            except (Empty, ValueError):  # ValueError on negative timeout
                return

    def _serve_forever() -> NoReturn:
        while True:
            if batch := [*_fetch_batch()]:
                fs, items = zip(*batch)
                _batch_invoke(partial(func, items), fs)
            else:
                sleep(0.001)

    def wrapper(items):
        fs_iter = iter(Future, None)
        fs = [f for x, f in zip(items, fs_iter) if not buf.put((f, x))]
        end_time = monotonic() + timeout
        return [f.result(end_time - monotonic()) for f in fs]

    Thread(target=_serve_forever, daemon=True).start()
    return update_wrapper(wrapper, func)
