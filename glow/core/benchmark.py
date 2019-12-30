__all__ = ('timer', 'time_this')

import atexit
import contextlib
import functools
import threading
import time
from dataclasses import dataclass
from typing import DefaultDict


@contextlib.contextmanager
def timer(name='Task', out: dict = None):
    def commit(start):
        duration = time.perf_counter() - start
        if out is not None:
            out[name] = duration
        print(f'{name} done in {duration:.4g} seconds')

    with contextlib.ExitStack() as stack:
        stack.callback(commit, time.perf_counter())
        yield


@dataclass
class Stat:
    call_count: int = 0
    duration: float = 0


def time_this(fn):
    module_import = time.perf_counter()
    lock = threading.RLock()
    stats = DefaultDict[str, Stat](Stat)

    def finalize():
        if not stats:
            return

        call_count = sum(v.call_count for v in stats.values())
        total_time = sum(v.call_count * v.duration for v in stats.values())
        total_runtime = time.perf_counter() - module_import + 1e-7

        print(f'{fn.__module__}:{fn.__qualname__} -'
              f' calls: {call_count},'
              f' total: {total_time:.4f},'
              f' mean: {total_time / call_count:.4f}'
              f' ({100*total_time/total_runtime:.2f}% of module),'
              f' threads: {len(stats)}')

    def commit(start: float, stat: Stat):
        duration = time.perf_counter() - start
        stat.call_count += 1
        stat.duration += (duration - stat.duration) / stat.call_count

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with lock:
            stat = stats[threading.get_ident()]

        with contextlib.ExitStack() as stack:
            stack.callback(commit, time.perf_counter(), stat)
            return fn(*args, **kwargs)

    wrapper.finalize = atexit.register(finalize)
    return wrapper
