__all__ = ('timer', 'time_this')

import contextlib
import functools
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Optional


@contextlib.contextmanager
def timer(name: str = 'Task',
          callback: Optional[Callable[[float], Any]] = None):
    if callback is None:
        def callback(duration: float) -> None:
            print(f'{name} done in {duration:.4g} seconds')

    assert callback is not None
    with contextlib.ExitStack() as stack:
        stack.callback(
            lambda init: callback(time.perf_counter() - init),  # type: ignore
            time.perf_counter(),
        )
        yield


@dataclass
class _ThreadInfo:
    count: int = 0
    duration: float = 0

    def update(self, start: float) -> None:
        duration = time.perf_counter() - start
        self.count += 1
        self.duration += (duration - self.duration) / self.count


def time_this(fn):
    lock = threading.RLock()
    infos = DefaultDict[str, _ThreadInfo](_ThreadInfo)

    def finalize(start):
        if not infos:
            return

        call_count = sum(i.count for i in infos.values())
        total_time = sum(i.count * i.duration for i in infos.values())
        total_runtime = time.perf_counter() - start + 1e-7

        print(f'{fn.__module__}:{fn.__qualname__} -'
              f' calls: {call_count},'
              f' total: {total_time:.4f},'
              f' per-call: {total_time / call_count:.4f}'
              f' ({100 * total_time / total_runtime:.2f}% of module),'
              f' threads: {len(infos)}')

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with lock:
            info = infos[threading.get_ident()]

        with contextlib.ExitStack() as stack:
            stack.callback(info.update, time.perf_counter())
            return fn(*args, **kwargs)

    wrapper.finalize = weakref.finalize(fn, finalize, time.perf_counter())
    return wrapper
