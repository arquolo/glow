__all__ = ['memprof', 'time_this', 'timer']

import atexit
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from itertools import accumulate, count
from threading import get_ident
from time import perf_counter_ns, process_time_ns, thread_time_ns
from typing import TYPE_CHECKING, Protocol, Self

from wrapt import ObjectProxy

from ._debug import whereami
from ._repr import si, si_bin

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


class _Nlwp:
    __slots__ = ('_add_event', '_get_max')

    def __init__(self) -> None:
        events = deque[int]()
        self._add_event = events.append

        deltas = iter(events.popleft, None)
        totals = accumulate(deltas)
        maximums = accumulate(totals, max, initial=0)
        self._get_max = maximums.__next__

    def __enter__(self) -> None:
        self._add_event(+1)
        self._get_max()

    def __exit__(self, *args) -> None:
        self._add_event(-1)
        self._get_max()

    def max(self) -> int:
        self._add_event(0)
        return self._get_max()


class _Stat:
    def __init__(self) -> None:
        self.calls = count()
        self.nlwp = _Nlwp()
        self.cpu_ns = _Times()
        self.all_ns = _Times()

    def __call__[
        **P, R
    ](self, op: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        with (
            self.nlwp,
            timer(self.all_ns.add),
            timer(self.cpu_ns.add, thread_time_ns),
        ):
            return op(*args, **kwargs)

    def stat(self) -> tuple[float, float, str] | None:
        if not (n := next(self.calls)):
            return None
        w = self.nlwp.max()
        t_ns = self.cpu_ns.total()  # CPU
        i_ns = self.all_ns.total() - t_ns  # idle = total - CPU
        t, i = t_ns / 1e9, i_ns / 1e9

        tail = (
            (f'{n} x {si(t / n)}s' + (f' @ {w}T' if w > 1 else ''))
            if n > 1
            else ''
        )
        return t, i, tail


class _Apply(Protocol):
    calls: count

    def __call__[
        **P, R
    ](self, fn: Callable[P, R], /, *args: P.args, **kwds: P.kwargs) -> R: ...


class _Proxy[T](ObjectProxy):
    __wrapped__: T
    _self_wrapper: _Apply

    def __init__(self, wrapped: T, wrapper: _Apply) -> None:
        super().__init__(wrapped)
        self._self_wrapper = wrapper


class _TimedCall[**P, R](_Proxy[Callable[P, R]]):
    def __get__(
        self, instance: object, owner: type | None
    ) -> '_BoundTimedCall':
        fn = self.__wrapped__.__get__(instance, owner)
        return _BoundTimedCall(fn, self._self_wrapper)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        next(self._self_wrapper.calls)
        r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        if isinstance(r, Generator):
            return _TimedGen(r, self._self_wrapper)
        if isinstance(r, Iterator):
            return _TimedIter(r, self._self_wrapper)
        return r


class _BoundTimedCall[**P, R](_TimedCall[P, R]):
    def __get__(self, instance: object, owner: type | None) -> Self:
        return self


class _TimedIter[Y](_Proxy[Iterator[Y]]):
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:
        return self._self_wrapper(self.__wrapped__.__next__)


class _TimedGen[Y, S, R](_Proxy[Generator[Y, S, R]]):
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Y:
        return self._self_wrapper(self.__wrapped__.__next__)

    def send(self, value: S, /) -> Y:
        return self._self_wrapper(self.__wrapped__.send, value)

    def throw(self, value: BaseException, /) -> Y:
        return self._self_wrapper(self.__wrapped__.throw, value)

    def close(self) -> None:
        return self._self_wrapper(self.__wrapped__.close)


# Wall time, i.e. sum of per-thread times, excluding sleep
_start = process_time_ns()
_stats = defaultdict[str, _Stat](_Stat)


@atexit.register
def _print_stats(*names: str) -> None:
    all_busy = (process_time_ns() - _start + 1) / 1e9

    stats = []
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
    return _TimedCall(fn, _stats[name])


time_this.finalizers = {}
