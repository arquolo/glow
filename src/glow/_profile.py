__all__ = ['memprof', 'time_this', 'timer']

import atexit
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from itertools import count
from threading import get_ident
from time import perf_counter_ns, process_time_ns, thread_time_ns
from typing import TYPE_CHECKING

from ._debug import whereami
from ._repr import si, si_bin
from ._streams import Stream, maximum_cumsum
from ._wrap import wrap

if TYPE_CHECKING:
    import psutil

    _THIS: psutil.Process | None

_THIS = None


@contextmanager
def memprof(
    name_or_callback: str | Callable[[float], object] | None = None, /
) -> Iterator[None]:
    global _THIS  # noqa: PLW0603
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
def _timer_callback(
    callback: Callable[[int], object],
    time: Callable[[], int] = perf_counter_ns,
    /,
) -> Iterator[None]:
    begin = time()
    try:
        yield
    finally:
        callback(time() - begin)


@contextmanager
def _timer_print(
    name: str | None = None, time: Callable[[], int] = perf_counter_ns, /
) -> Iterator[None]:
    begin = time()
    try:
        yield
    finally:
        end = time()
        name = name or f'{whereami(2, 1)} line'
        print(f'{name} done in {si((end - begin) / 1e9)}s')


def timer(
    name_or_callback: str | Callable[[int], object] | None = None,
    time: Callable[[], int] = perf_counter_ns,
    /,
    *,
    disable: bool = False,
) -> AbstractContextManager[None]:
    if disable:
        return nullcontext()
    if callable(name_or_callback):
        return _timer_callback(name_or_callback, time)
    return _timer_print(name_or_callback, time)


def _to_fname(obj) -> str:
    if not hasattr(obj, '__module__') or not hasattr(obj, '__qualname__'):
        obj = type(obj)
    if obj.__module__ == 'builtins':
        return obj.__qualname__
    return f'{obj.__module__}.{obj.__qualname__}'


class _Times(dict[int, int]):
    def add(self, value: int) -> None:
        idx = get_ident()
        self[idx] = self.get(idx, 0) + value

    def total(self) -> int:
        return sum(self.values())


@dataclass(frozen=True, slots=True)
class _Stat:
    calls: count = field(default_factory=count)
    reads: count = field(default_factory=count)
    active_calls: Stream[int, int] = field(default_factory=maximum_cumsum)
    busy_ns: _Times = field(default_factory=_Times)
    idle_ns: _Times = field(default_factory=_Times)

    def __call__[**P, R](
        self, op: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        self.active_calls.send(+1)
        # Real time (wall-clock time) - total elapsed time including
        # waiting for I/O (like time.sleep, lock.acquire, e.t.c.).
        total = perf_counter_ns()
        # Time spent executing this thread, including kernel time.
        active = thread_time_ns()
        try:
            return op(*args, **kwargs)
        finally:
            active = thread_time_ns() - active
            total = perf_counter_ns() - total
            idle = max(0, total - active)
            self.busy_ns.add(active)
            self.idle_ns.add(idle)
            self.active_calls.send(-1)

    def stat(self) -> tuple[float, float, str] | None:
        if not (n := next(self.calls) - next(self.reads)):
            return None
        w = self.active_calls.send(0)
        t_ns = self.busy_ns.total()  # CPU = D(thread_time)
        i_ns = self.idle_ns.total()  # idle = D(perf_counter) - D(thread_time)
        t, i = t_ns / 1e9, i_ns / 1e9

        tail = (
            (f'{n} x {si(t / n)}s' + (f' @ {w}T' if w > 1 else ''))
            if n > 1
            else ''
        )
        return t, i, tail


# Wall time, i.e. sum of per-thread times, excluding sleep
_start = process_time_ns()
_stats = defaultdict[str, _Stat](_Stat)


@atexit.register
def _print_stats(*names: str) -> None:
    all_busy = (process_time_ns() - _start + 1) / 1e9

    stats: list[tuple[float, float, str, str]] = []  # (busy, idle, tail, name)
    names = names or tuple(_stats)
    for name in names:
        if not (stat := _stats.pop(name, None)):
            continue
        if not (lines := stat.stat()):
            continue
        stats.append((*lines, name))

    for busy, idle, tail, name in sorted(stats):
        print(
            f'{busy / all_busy:6.2%}',
            f'{si(busy):>5s}s + {si(idle):>5s}s',
            name,
            tail,
            sep=' - ',
        )


def time_this(fn=None, /, *, name: str | None = None, disable: bool = False):
    """Log function and/or generator timings at program exit"""
    if fn is None:
        return partial(time_this, name=name, disable=disable)
    if disable:
        return fn

    if name is None:
        name = _to_fname(fn)

    time_this.finalizers[fn] = fn.log_timing = partial(_print_stats, name)
    return wrap(fn, _stats[name])


time_this.finalizers = {}
