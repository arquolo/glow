__all__ = ['mapped']

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

from ._reduction import reducers, serialize
from .len_helpers import SizedIterable, as_sized
from .more import chunked

_T = TypeVar('_T')
_R = TypeVar('_R')
_NUM_CPUS = os.cpu_count() or 0
_IDLE_WORKER_TIMEOUT = 10

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


def _get_pool(workers: int) -> Executor:
    return loky.get_reusable_executor(  # type: ignore
        workers,
        timeout=_IDLE_WORKER_TIMEOUT,
        job_reducers=reducers,
        result_reducers=reducers,
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

    def submit_with_callback() -> Iterator['Future[_T]']:
        for f in fs_submit:
            f.add_done_callback(done.put)
            yield f

    fs_submit_ = submit_with_callback()
    fs.update(islice(fs_submit_, latency))
    while fs:
        yield (fut := done.get()).result()
        fs.remove(fut)
        fs.update(islice(fs_submit_, 1))


def mapped(fn: Callable[..., _R],
           *iterables: Iterable[_T],
           workers: int = _NUM_CPUS,
           latency: int = 2,
           chunk_size: int = 0,
           ordered: bool = True) -> SizedIterable[_R]:
    """Returns an iterator equivalent to map(fn, *iterables).

    Differences:
    - Uses multiple threads or processes, whether chunks_size is zero or not.
    - Unlike multiprocessing.Pool or concurrent.futures.Executor
      *almost* never deadlocks on any exception or Ctrl-C interruption.

    Parameters:
    - fn - A callable that will take as many arguments as there are passed
      iterables.
    - workers - Count of workers, by default all hardware threads are occupied.
    - latency - Count of tasks each worker can grab.
    - chunk_size - The size of the chunks the iterable will be broken into
      before being passed to a worker. Zero disables multiprocessing.

    Calls may be evaluated out-of-order.
    """
    if workers == 0:
        return cast(SizedIterable[_R], map(fn, *iterables))

    terminator = None
    stack = contextlib.ExitStack()
    if chunk_size:
        pool = _get_pool(workers)
        terminator = atexit.register(pool.shutdown, kill_workers=True)
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
        if terminator is not None:
            atexit.unregister(terminator)

    return iter_results()
