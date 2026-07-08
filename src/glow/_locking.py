__all__ = [
    'AbsEvent',
    'AbsManager',
    'AbsQueue',
    'f_exception',
    'f_result',
    'q_get',
]

import sys
from concurrent.futures import Future
from queue import Empty
from time import sleep
from typing import Protocol

from ._types import Some

_PERIOD = 0.01


class AbsManager(Protocol):
    def Event(self) -> 'AbsEvent': ...  # noqa: N802
    def Queue(self, /, maxsize: int) -> 'AbsQueue': ...  # noqa: N802


class AbsQueue[T](Protocol):
    def get(self, block: bool = ..., timeout: float | None = ...) -> T: ...
    def put(self, item: T) -> None: ...


class AbsEvent(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...


def f_result[T](f: Future[T], cancel: bool = True) -> Some[T] | BaseException:
    try:
        return exc if (exc := f_exception(f)) else Some(f.result())
    finally:
        if cancel:
            f.cancel()
        del f


if sys.platform == 'win32':

    def f_exception[T](f: Future[T], /) -> BaseException | None:
        # See issues
        # https://bugs.python.org/issue29971
        # https://github.com/dask/dask/pull/2144#issuecomment-290556996
        # https://github.com/dask/dask/pull/2144/files
        # https://github.com/python/cpython/issues/74157
        # FIXED in py3.15+
        while True:
            try:
                return f.exception(timeout=_PERIOD)
            except TimeoutError:
                sleep(0)  # Force switch to another thread to proceed

    def q_get[T](q: AbsQueue[T]) -> T:
        # On Windows lock.acquire called without a timeout is not interruptible
        # See issues
        # https://bugs.python.org/issue29971
        # https://github.com/dask/dask/pull/2144#issuecomment-290556996
        # https://github.com/dask/dask/pull/2144/files
        # https://github.com/python/cpython/issues/74157
        # FIXED in py3.15+
        while True:
            try:
                return q.get(timeout=_PERIOD)
            except Empty:
                sleep(0)  # Force switch to another thread to proceed

else:
    f_exception = Future.exception

    def q_get[T](q: AbsQueue[T]) -> T:
        return q.get()
