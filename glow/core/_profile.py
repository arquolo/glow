from __future__ import annotations

__all__ = ['memprof', 'time_this', 'timer']

import functools
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wrapt import ObjectProxy

from ._repr import si, si_bin
from .debug import whereami

if TYPE_CHECKING:
    import psutil
    _THIS: psutil.Process | None

_THIS = None


@contextmanager
def memprof(name_or_callback: str | Callable[[float], object] | None = None,
            /) -> Iterator[None]:
    global _THIS
    if _THIS is None:
        import psutil
        _THIS = psutil.Process()

    init = _THIS.memory_info().rss
    try:
        yield
    finally:
        size = _THIS.memory_info().rss - init
        if callable(name_or_callback):
            name_or_callback(size)
        else:
            name = name_or_callback
            if name is None:
                name = f'{whereami(2, 1)} line'
            sign = '+' if size >= 0 else ''
            print(f'{name} done: {sign}{si_bin(size)}')


@contextmanager
def timer(name_or_callback: str | Callable[[float], object] | None = None,
          /) -> Iterator[None]:
    init = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - init
        if callable(name_or_callback):
            name_or_callback(duration)
        else:
            name = name_or_callback
            if name is None:
                name = f'{whereami(2, 1)} line'
            print(f'{name} done in {si(duration)}s')


def _to_fname(obj) -> str:
    if not hasattr(obj, '__module__') or not hasattr(obj, '__qualname__'):
        obj = type(obj)
    if obj.__module__ == 'builtins':
        return obj.__qualname__
    return f'{obj.__module__}.{obj.__qualname__}'


@dataclass
class _Stat:
    calls: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    times: dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def update_call(self, duration: float):
        # TODO: drop len(set[ident]), use max(len(set[ident])),
        # TODO:  aka max concurrency
        idx = threading.get_ident()
        self.calls[idx] += 1
        self.times[idx] += duration

    def update_next(self, duration: float):
        idx = threading.get_ident()
        self.times[idx] += duration

    def stat(self) -> tuple[str, float] | None:
        if not self.calls:
            return None

        w = len(self.calls | self.times)
        n = sum(self.calls.values())
        t = sum(self.times.values())
        tail = (f' ({n} x {si(t / n)}s)' if w == 1 else
                f' ({n} x {si(t / n)}s @ {w}T)') if n > 1 else ''
        return f'{si(t)}s{tail}', t


_start = time.perf_counter()
_stats: dict[str, _Stat] = defaultdict(_Stat)
_lock = threading.RLock()


class _TimedIter(ObjectProxy):
    def __init__(self, wrapped, callback):
        super().__init__(wrapped)
        self._self_callback = callback

    def __iter__(self):
        return self

    def __next__(self):
        with timer(self._self_callback):
            return next(self.__wrapped__)

    def close(self):
        with timer(self._self_callback):
            return self.__wrapped__.close()

    def send(self, value):
        with timer(self._self_callback):
            return self.__wrapped__.send(value)

    def throw(self, typ, val=None, tb=None):
        with timer(self._self_callback):
            return self.__wrapped__.throw(typ, val, tb)


def _print_stats(name: str):
    with _lock:
        if not (stat := _stats.pop(name)):
            return
    if not (lines := stat.stat()):
        return

    text, runtime = lines
    total = time.perf_counter() - _start + 1e-7
    print(f'{name} - {text} - {runtime / total:.2%} of all')


def time_this(fn=None, /, *, name: str | None = None):
    """Log function and/or generator timings at program exit"""
    if fn is None:
        return functools.partial(time_this, name=name)

    if name is None:
        name = _to_fname(fn)

    stat = _stats[name]
    call_cbk, next_cbk = stat.update_call, stat.update_next

    def wrapper(*args, **kwargs):
        with timer(call_cbk):
            res = fn(*args, **kwargs)
        return _TimedIter(res, next_cbk) if isinstance(res, Iterator) else res

    wrapper.finalize = weakref.finalize(fn, _print_stats, name)  # type: ignore
    return functools.update_wrapper(wrapper, fn)
