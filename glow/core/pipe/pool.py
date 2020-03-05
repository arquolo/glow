__all__ = ('mapped', )

import atexit
import contextlib
import os
import queue
import signal
from concurrent.futures import Future, ThreadPoolExecutor
from cProfile import Profile
from itertools import chain, islice
from pstats import Stats
from typing import Callable, Deque, Iterable, Iterator, Set, TypeVar, cast

import loky

from ..reduction import dispatch_table_patches, serialize
from .len_helpers import SizedIterable, as_sized
from .more import chunked

_T = TypeVar('_T')
_R = TypeVar('_R')
_NUM_CPUS = os.cpu_count()
_GC_TIMEOUT = 10

loky.backend.context.set_start_method('loky_init_main')


def _mp_profile():
    prof = Profile()
    prof.enable()

    def _finalize(lines=50):
        prof.disable()
        with open(f'prof-{os.getpid()}.txt', 'w') as fp:
            Stats(prof, stream=fp).sort_stats('cumulative').print_stats(lines)

    atexit.register(_finalize)


def _initializer():
    # `signal.signal` suppresses KeyboardInterrupt in child processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get('_GLOW_MP_PROFILE'):
        _mp_profile()


def _get_pool(workers):
    return loky.get_reusable_executor(
        workers,
        timeout=_GC_TIMEOUT,
        job_reducers=dispatch_table_patches,
        result_reducers=dispatch_table_patches,
        initializer=_initializer,
    )


def _reduce_ordered(fs_submit: Iterator['Future[_T]'],
                    stack: contextlib.ExitStack, latency: int) -> Iterator[_T]:
    fs = Deque['Future[_T]']()
    stack.callback(lambda: {fut.cancel() for fut in reversed(fs)})

    fs.extend(islice(fs_submit, latency))
    while fs:
        yield fs.popleft().result()
        fs.extend(islice(fs_submit, 1))


def _reduce_completed(fs_submit: Iterator['Future[_T]'],
                      stack: contextlib.ExitStack,
                      latency: int) -> Iterator[_T]:
    fs: Set['Future[_T]'] = set()
    stack.callback(lambda: {fut.cancel() for fut in fs})
    done: 'queue.Queue[Future[_T]]' = queue.Queue()

    def submit() -> Iterator['Future[_T]']:
        for f in fs_submit:
            f.add_done_callback(done.put)
            yield f

    fs_submit_ = submit()
    fs.update(islice(fs_submit_, latency))
    while fs:
        fut = done.get()
        yield fut.result()
        fs.remove(fut)
        fs.update(islice(fs_submit_, 1))


def mapped(fn: Callable[..., _R],
           *iterables: Iterable[_T],
           workers=_NUM_CPUS,
           latency=2,
           chunk_size=0,
           ordered=True) -> SizedIterable[_R]:
    """
    Concurrently applies `fn` callable to each element in zipped `iterables`.
    Keeps order if nessessary. Never hang. Friendly to CTRL+C.
    Uses all processing power by default.

    Parameters:
      - `workers` - count of workers
        (default: same as `os.cpu_count()`)
      - `latency` - count of tasks each workers can grab
        (default: `2`)
      - `chunk_size` - size of chunk to pass to each `Process`, if not `0`
        (default: `0`)
      - `ordered` - if disabled, yields items in order of completion
        (default: `True`)
    """
    if workers == 0:
        return cast(SizedIterable[_R], map(fn, *iterables))

    stack = contextlib.ExitStack()
    if chunk_size:
        pool = _get_pool(workers)
        proxy = serialize(fn)
    else:
        pool = stack.enter_context(ThreadPoolExecutor(workers))
        proxy = serialize(fn, mp=False)
        chunk_size = 1

    latency += workers
    reducer = _reduce_ordered if ordered else _reduce_completed

    @as_sized(hint=lambda: min(map(len, iterables)))  # type: ignore
    def iter_results():
        with stack:
            iterable = chunked(zip(*iterables), chunk_size)
            fs = (pool.submit(proxy, *item) for item in iterable)
            yield from chain.from_iterable(reducer(fs, stack, latency))

    return iter_results()
