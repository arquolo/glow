from __future__ import annotations

__all__ = ['streaming', 'call_once', 'threadlocal', 'shared_call']

import threading
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic
from typing import Any, TypeVar, cast
from weakref import WeakValueDictionary

from .._thread_quota import ThreadQuota
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
class _UFuture:
    _fn: Callable[[], Any]
    _lock: Lock = field(default_factory=Lock)
    _result: Any = _unset
    _exception: BaseException | None = None

    def result(self):
        with self._lock:
            if self._exception:
                raise self._exception
            if self._result is not _unset:
                return self._result

            try:
                self._result = r = self._fn()
                return r
            except BaseException as e:
                self._exception = e
                raise


def call_once(fn: _ZeroArgsF) -> _ZeroArgsF:
    """Makes `fn()` callable a singleton.
    DO NOT USE with recursive functions"""
    def wrapper():
        return f.result()

    fn._future = f = _UFuture(fn)  # type: ignore
    return cast(_ZeroArgsF, update_wrapper(wrapper, fn))


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one.
    DO NOT USE with recursive functions"""
    fs = WeakValueDictionary[Hashable, _UFuture]()
    lock = Lock()

    def wrapper(*args, **kwargs):
        key = make_key(*args, **kwargs)

        with lock:  # Create only one task per args-kwargs set
            if not (f := fs.get(key)):
                fs[key] = f = _UFuture(partial(fn, *args, **kwargs))

        return f.result()

    return cast(_F, update_wrapper(wrapper, fn))


def _fetch_batch(q: SimpleQueue[_T], batch_size: int,
                 timeout: float) -> list[_T]:
    # Wait indefinitely for the first item
    batch = [q.get()]

    # Waiting is limited only for the consecutive items
    end = monotonic() + timeout
    while len(batch) < batch_size and (left := end - monotonic()) > 0:
        try:
            batch.append(q.get(timeout=left))
        except Empty:
            break
    return batch


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


def _compute(func, batch):
    fs, items = zip(*batch)
    _batch_invoke(partial(func, items), fs)


def streaming(func=None, *, batch_size, timeouts=(0.1, 20.), workers=1):
    """
    Delays start of computation to until batch is collected.
    Accepts two timeouts (in seconds):
    - first controls waiting of items from the consecutive calls.
    - second is responsible for waiting of results.

    Simplified version of https://github.com/ShannonAI/service-streamer

    Note: currently supports only functions and bound methods.
    """
    if func is None:
        return partial(streaming, batch_size=batch_size, timeouts=timeouts)

    assert callable(func)
    assert workers >= 1
    q = SimpleQueue()
    batch_timeout, result_timeout = timeouts
    executor = ThreadQuota(workers)

    def _collect():
        while True:
            batch = _fetch_batch(q, batch_size, batch_timeout)
            executor.submit(_compute, func, batch)

    Thread(target=_collect, daemon=True).start()

    def wrapper(items):
        fs = {Future(): item for item in items}
        for f_x in fs.items():
            q.put(f_x)

        if wait(fs, result_timeout, return_when='FIRST_EXCEPTION').not_done:
            raise TimeoutError
        return [f.result() for f in fs]

    # TODO: if func is instance method - recreate wrapper per instance
    # TODO: find how to distinguish between
    # TODO:  not yet bound method and plain function
    # TODO:  maybe implement __get__ on wrapper
    return update_wrapper(wrapper, func)
