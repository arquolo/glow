from __future__ import annotations

__all__ = ['memprof', 'time_this', 'timer']

import functools
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from ._repr import si, si_bin
from .debug import whereami

_F = TypeVar('_F', bound=Callable)

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


def time_this(fn: _F) -> _F:
    """Log function timings at program exit"""
    infos: dict[int, list] = {}

    def finalize(start):
        if not infos:
            return

        num_calls = sum(num for num, _ in infos.values())
        total_time = sum(num * duration for num, duration in infos.values())
        total_runtime = time.perf_counter() - start + 1e-7

        print(f'{fn.__module__}:{fn.__qualname__} -'
              f' calls: {num_calls},'
              f' total: {si(total_time)}s,'
              f' per-call: {si(total_time / num_calls)}s'
              f' ({100 * total_time / total_runtime:.2f}% of module),'
              f' threads: {len(infos)}')

    def callback(duration: float):
        info = infos.setdefault(threading.get_ident(), [0, 0.])
        info[0] += 1
        info[1] += (duration - info[1]) / info[0]

    @timer(callback)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    finalizer = weakref.finalize(fn, finalize, time.perf_counter())
    wrapper.finalize = finalizer  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, fn))
