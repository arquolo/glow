__all__ = [
    'AbsEvent',
    'AbsManager',
    'AbsQueue',
    'maybe_future',
    'q_get',
]

import sys
from concurrent.futures import CancelledError, Future
from queue import Empty
from time import sleep
from typing import Protocol, cast

from ._dev import hide_frame
from ._types import Maybe

# On Windows lock.acquire called without a timeout is not interruptible
# See issues
# https://bugs.python.org/issue29971
# https://github.com/dask/dask/pull/2144#issuecomment-290556996
# https://github.com/dask/dask/pull/2144/files
# https://github.com/python/cpython/issues/74157
# FIXED in py3.15+
if sys.platform == 'win32':
    _SWITCH_INTERVAL = 0.01  # or sys.getswitchinterval() which is 0.005
else:
    _SWITCH_INTERVAL = None


class AbsQueue[T](Protocol):
    def get(self, block: bool = ..., timeout: float | None = ...) -> T: ...
    def put(self, item: T) -> None: ...


class AbsEvent(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class AbsManager(Protocol):
    def Event(self) -> AbsEvent: ...  # noqa: N802
    def Queue(self, /, maxsize: int) -> AbsQueue: ...  # noqa: N802


def maybe_future[T](f: Future[T], cancel: bool = False) -> Maybe[T]:
    """Version with single acquire-release call of this:
    >>> try:
    ...     if (exc := f.exception()) is not None:  # <- acquire-release
    ...         return exc
    ...     return [f.result()]  # <- acquire-release
    ... except:
    ...     if cancel:
    ...         f.cancel()  # <- acquire-release
    ...     del f
    ...     raise
    """
    invoke_callbacks = False
    try:
        with f._condition, hide_frame:
            try:
                while f._state in ['PENDING', 'RUNNING']:
                    with hide_frame:
                        f._condition.wait(_SWITCH_INTERVAL)
            except:
                if cancel and f._state == 'PENDING':
                    f._state = 'CANCELLED'
                    f._condition.notify_all()
                    invoke_callbacks = True
                raise
            else:
                if f._state in ['CANCELLED', 'CANCELLED_AND_NOTIFIED']:
                    raise CancelledError
                if f._exception is not None:
                    return f._exception
                return [cast('T', f._result)]
    except:
        if invoke_callbacks:
            f._invoke_callbacks()  # type: ignore[attr-defined]
        raise
    finally:
        del f


if sys.platform == 'win32':

    def q_get[T](q: AbsQueue[T]) -> T:
        while True:
            try:
                return q.get(timeout=_SWITCH_INTERVAL)
            except Empty:
                sleep(0)  # Force switch to another thread to proceed

else:

    def q_get[T](q: AbsQueue[T]) -> T:
        return q.get()
