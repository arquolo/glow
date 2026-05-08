__all__ = [
    'memprof',
    'time_this',
    'timer',
    'whereami',
]

import atexit
import gc
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from inspect import currentframe, getmodule, isfunction
from itertools import count, islice
from time import perf_counter_ns, process_time_ns, thread_time_ns
from typing import TYPE_CHECKING

from ._cache import memoize
from ._repr import si, si_bin
from ._streams import Stream, cumsum, maximum_cumsum
from ._types import Callback, Get
from ._wrap import wrap

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType


if TYPE_CHECKING:
    import psutil

    _THIS: psutil.Process | None

_THIS = None

# ---------------------------------- memory ----------------------------------


@contextmanager
def memprof(
    name_or_callback: str | Callback[float] | None = None, /
) -> Iterator[None]:
    global _THIS  # noqa: PLW0603
    if _THIS is None:
        import psutil  # noqa: PLC0415

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


# ----------------------------------- time -----------------------------------


@contextmanager
def _timer_callback(
    callback: Callback[int], time: Get[int] = perf_counter_ns, /
) -> Iterator[None]:
    begin = time()
    try:
        yield
    finally:
        callback(time() - begin)


@contextmanager
def _timer_print(
    name: str | None = None, time: Get[int] = perf_counter_ns, /
) -> Iterator[None]:
    begin = time()
    try:
        yield
    finally:
        end = time()
        name = name or f'{whereami(2, 1)} line'
        print(f'{name} done in {si((end - begin) / 1e9)}s')


def timer(
    name_or_callback: str | Callback[int] | None = None,
    time: Get[int] = perf_counter_ns,
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


@dataclass(frozen=True, slots=True)
class _Profiler:
    # Call count
    calls: count = field(default_factory=count)
    # Statistics requests count
    reads: count = field(default_factory=count)
    # Parallelism/concurrency
    active_calls: Stream[int, int] = field(default_factory=maximum_cumsum)
    # Time spent executing this thread, including kernel time, but not I/O.
    busy_ns: Stream[int, int] = field(default_factory=cumsum)
    # Idle time - elapsed I/O time (like time.sleep, lock.acquire, e.t.c.).
    idle_ns: Stream[int, int] = field(default_factory=cumsum)

    def suspend(self) -> Get[None]:
        self.idle_ns.send(-perf_counter_ns())
        return self.resume

    def resume(self) -> None:
        self.idle_ns.send(+perf_counter_ns())

    def __call__[**P, R](
        self, op: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        self.active_calls.send(+1)
        t_cpu = thread_time_ns()
        self.idle_ns.send(+t_cpu - perf_counter_ns())
        self.busy_ns.send(-t_cpu)
        try:
            return op(*args, **kwargs)
        finally:
            t_cpu = thread_time_ns()
            self.busy_ns.send(+t_cpu)
            self.idle_ns.send(-t_cpu + perf_counter_ns())
            self.active_calls.send(-1)

    def stat(self) -> tuple[float, float, str] | None:
        if not (n := next(self.calls) - next(self.reads)):
            return None
        concurrency = self.active_calls.send(0)
        idle = self.idle_ns.send(0) / 1e9  # Wall-clock - user+sys
        busy = self.busy_ns.send(0) / 1e9  # user+system time

        tail = ''
        if n > 1:
            tail += f'{n} x {si(busy / n)}s'
            if concurrency > 1:
                tail += f' @ {concurrency}T'
        return busy, idle, tail


# Wall time, i.e. sum of per-thread times, excluding sleep
_start = process_time_ns()
_profilers = defaultdict[str, _Profiler](_Profiler)


@atexit.register
def _print_stats(*names: str) -> None:
    all_busy = (process_time_ns() - _start + 1) / 1e9

    stats: list[tuple[float, float, str, str]] = []  # (busy, idle, tail, name)
    names = names or tuple(_profilers)
    for name in names:
        if not (profiler := _profilers.pop(name, None)):
            continue
        if not (lines := profiler.stat()):
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
    """Log function and/or generator timings at program exit."""
    if fn is None:
        return partial(time_this, name=name, disable=disable)
    if disable:
        return fn

    if name is None:
        name = _to_fname(fn)

    fin = fn.log_timing = partial(_print_stats, name)
    time_this.finalizers[fn] = fin  # type: ignore[attr-defined]
    return wrap(fn, _profilers[name])


time_this.finalizers = {}  # type: ignore[attr-defined]


# --------------------------------- location ---------------------------------


def _frame_hash(frame: 'FrameType') -> tuple[str, int]:
    return frame.f_code.co_filename, frame.f_lineno


@memoize(100, policy='lru', key_fn=_frame_hash)
def _get_source(frame: 'FrameType') -> str:
    # Get source module name
    modname = (
        spec.name
        if (module := getmodule(frame)) and (spec := module.__spec__)
        else '__main__'
    )

    # Get source code name (method or function name)
    code = frame.f_code
    codename = next(
        (f.__qualname__ for f in gc.get_referrers(code) if isfunction(f)),
        code.co_name,
    )
    if codename == '<module>':
        codename = ''

    return f'{modname}:{codename}:{frame.f_lineno}'


def _get_source_calls(frame: 'FrameType | None') -> 'Iterator[str]':
    while frame:
        yield _get_source(frame)
        if frame.f_code.co_name == '<module>':  # Stop on module-level scope
            return
        frame = frame.f_back


def stack(skip: int = 0, limit: int | None = None) -> 'Iterator[str]':
    """Return iterator of FrameInfos, stopping on module-level scope."""
    frame = currentframe()
    calls = _get_source_calls(frame)
    calls = islice(calls, skip + 1, None)  # Skip 'skip' outerless frames
    if not limit:
        return calls
    if limit < 0:
        return iter(list(calls)[:limit])
    return islice(calls, limit)  # Keep at most `limit` outer frames


def whereami(skip: int = 0, limit: int | None = None) -> str:
    calls = stack(skip + 1, limit)
    return ' -> '.join(reversed([*calls]))
