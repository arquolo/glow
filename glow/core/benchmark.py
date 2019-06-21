__all__ = 'timer', 'time_this'

import atexit
import functools
import time
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from dataclasses import dataclass
from threading import RLock, get_ident

from . import print_


@contextmanager
def timer(name='Task', out: dict = None):
    def commit(start):
        duration = time.perf_counter() - start
        if out is not None:
            out[name] = duration
        print_(f'{name} done in {duration:.4g} seconds')

    with ExitStack() as stack:
        stack.callback(commit, time.perf_counter())
        yield


@dataclass
class Stat:
    call_count: int = 0
    duration: float = 0


def time_this(fn):
    module_import = time.perf_counter()
    lock = RLock()
    stats = defaultdict(Stat)

    @atexit.register
    def finalize():
        if not stats:
            return

        call_count = sum(v.call_count for v in stats.values())
        total_time = sum(v.call_count * v.duration for v in stats.values())
        total_runtime = time.perf_counter() - module_import + 1e-7

        print_(f'{fn.__module__}:{fn.__qualname__} -'
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
            stat = stats[get_ident()]

        with ExitStack() as stack:
            stack.callback(commit, time.perf_counter(), stat)
            return fn(*args, **kwargs)

    return wrapper
