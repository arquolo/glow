__all__ = ('call_once', 'threadlocal', 'interpreter_lock', 'shared_call')

import contextlib
import sys
import functools
import threading
from concurrent.futures import Future
from typing import Callable, TypeVar, cast
from weakref import WeakValueDictionary

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)


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

    return cast(Callable[[], _T], wrapper)


@contextlib.contextmanager
def interpreter_lock(timeout=1_000):
    """
    Completely forbids thread switching in underlying scope.
    Thus makes it fully thread-safe, although adds high performance penalty.

    See tests for examples.
    """
    with contextlib.ExitStack() as stack:
        stack.callback(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(timeout)
        yield


class _Stack(contextlib.ExitStack):
    def defer(self, fn: Callable[..., _T], *args, **kwargs) -> 'Future[_T]':
        def apply(future: 'Future[_T]') -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        future: 'Future[_T]' = Future()
        self.callback(apply, future)
        return future


def call_once(fn: _F) -> _F:
    """Makes `fn()` callable a singleton"""
    lock = threading.RLock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _Stack() as stack:
            with lock:
                if fn.__future__ is None:
                    fn.__future__ = stack.defer(fn, *args, **kwargs)

        return fn.__future__.result()

    fn.__future__ = None  # type: ignore
    return cast(_F, wrapper)


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one"""
    lock = threading.RLock()
    futures: 'WeakValueDictionary[str, Future]' = WeakValueDictionary()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'

        with _Stack() as stack:
            with lock:
                try:
                    future = futures[key]
                except KeyError:
                    futures[key] = future = stack.defer(fn, *args, **kwargs)

        return future.result()

    return cast(_F, wrapper)
