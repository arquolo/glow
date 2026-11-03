import gc
import os
import pickle
import sys
import weakref
from concurrent.futures import Future
from contextlib import ExitStack, nullcontext
from functools import partial
from multiprocessing import Manager
from multiprocessing.shared_memory import SharedMemory
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
from loky import ProcessPoolExecutor

from glow import _parallel, _reduction


class ArrayLookup:
    def __init__(self, values, offsets):
        self.values = values
        self.offsets = offsets

    def __call__(self, index, extra=0):
        return int(self.values[index] + self.offsets[index] + extra)


def describe_lookup(lookup):
    return os.getpid(), lookup(3), lookup.call.values.flags.writeable


class GatedLookup:
    def __init__(self, gate, fail):
        self.values = np.arange(8)
        self.gate = gate
        self.fail = fail

    def __call__(self, index):
        if index == 0 and self.fail:
            raise ValueError('worker failed')
        if index and not self.gate.wait(timeout=20):
            raise TimeoutError('test gate was not released')
        return int(self.values[index])


def failing_source():
    yield 0
    yield 1
    raise ValueError('source failed')


@pytest.fixture(params=['mmap', 'shmem'])
def proxy_mode(request):
    return request.param


@pytest.fixture
def proxy_factory(proxy_mode, monkeypatch, tmp_path):
    # Keep POSIX backing files inside the test directory.
    monkeypatch.setattr(_reduction, '_SHM_ROOT', tmp_path)
    stack = ExitStack()

    def make(call):
        proxy = _reduction.move_to_shmem(call, mode=proxy_mode)
        return stack.enter_context(proxy)

    yield make

    # Reconstructed arrays must release their views before closing mappings.
    gc.collect()
    stack.close()
    gc.collect()


@pytest.fixture
def lookup():
    return ArrayLookup(np.arange(1024), np.arange(1024) * 10)


def test_proxy_without_buffers(proxy_factory):
    proxy = proxy_factory(partial(pow, 2))
    with pytest.raises(RuntimeError, match='parent process'):
        proxy(5)
    assert proxy.memos == []
    restored = pickle.loads(pickle.dumps(proxy))
    assert restored(5) == 32


def test_proxy_roundtrip(proxy_factory, lookup):
    proxy = proxy_factory(lookup)
    with pytest.raises(RuntimeError, match='parent process'):
        proxy(3, 7)
    assert len(proxy.memos) == 2
    payload = pickle.dumps(proxy)
    assert len(payload) < lookup.values.nbytes
    restored = pickle.loads(payload)
    assert restored is not proxy
    assert restored(3, 7) == 40
    np.testing.assert_array_equal(restored.call.values, lookup.values)
    np.testing.assert_array_equal(restored.call.offsets, lookup.offsets)
    assert restored.call.values.dtype == lookup.values.dtype
    assert restored.call.values.shape == lookup.values.shape
    with pytest.raises(RuntimeError, match='back to parent'):
        pickle.dumps(restored)


def test_proxy_bound_method(proxy_factory, lookup):
    proxy = proxy_factory(lookup.__call__)
    assert len(proxy.memos) == 2
    restored = pickle.loads(pickle.dumps(proxy))
    assert restored(3, 7) == 40


@pytest.mark.parametrize('order', ['C', 'F'])
def test_proxy_partial_with_array(proxy_factory, order):
    values = np.array(np.arange(12).reshape(3, 4), order=order)
    values.flags.writeable = False
    proxy = proxy_factory(partial(np.matmul, values))
    restored = pickle.loads(pickle.dumps(proxy))
    assert len(proxy.memos) == 1
    np.testing.assert_array_equal(
        restored(np.arange(4)), values @ np.arange(4)
    )


def test_proxy_noncontiguous_array(proxy_factory):
    values = np.arange(30).reshape(5, 6)[:, ::2]
    proxy = proxy_factory(partial(np.sum, values))
    restored = pickle.loads(pickle.dumps(proxy))
    assert restored() == values.sum()
    # NumPy serializes noncontiguous arrays in-band.
    assert proxy.memos == []


@pytest.mark.parametrize('shape', [(0,), (2, 0, 3)])
def test_proxy_empty_buffer(proxy_factory, shape):
    values = np.empty(shape, dtype=np.float64)
    proxy = proxy_factory(partial(np.sum, values))
    assert proxy.memos == []
    restored = pickle.loads(pickle.dumps(proxy))
    assert restored() == 0
    assert restored.call.args[0].shape == shape
    with ProcessPoolExecutor(max_workers=1) as executor:
        assert executor.submit(proxy).result(timeout=30) == 0


def test_proxy_mixed_empty_buffers(proxy_factory):
    proxy = proxy_factory(ArrayLookup(np.empty(0), np.arange(4)))
    assert len(proxy.memos) == 1
    restored = pickle.loads(pickle.dumps(proxy))
    assert restored.call.values.size == 0
    np.testing.assert_array_equal(restored.call.offsets, np.arange(4))


def test_shmem_parent_unlinks(monkeypatch):
    unlinked = []
    original_unlink = SharedMemory.unlink

    def unlink(sm):
        unlinked.append(sm.name)
        original_unlink(sm)

    monkeypatch.setattr(SharedMemory, 'unlink', unlink)
    proxy = _reduction.move_to_shmem(partial(np.sum, np.arange(4)), 'shmem')
    name = proxy.memos[0][0].name
    child = pickle.loads(pickle.dumps(proxy))
    del proxy
    gc.collect()
    assert unlinked == [name]
    # Existing attachments remain usable after the name is unlinked.
    assert child() == 6
    if os.name != 'nt':
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_shmem_unlink_already_removed():
    proxy = _reduction.move_to_shmem(partial(np.sum, np.arange(4)), 'shmem')
    sm, _ = proxy.memos[0]
    sm.unlink()
    _reduction._unlink_shmem(sm)


def test_shmem_view_outlives_owner(monkeypatch):
    errors = []
    monkeypatch.setattr(sys, 'unraisablehook', errors.append)
    sm = _reduction._SharedMemory(create=True, size=8)
    mapping = weakref.ref(sm._mmap)
    descriptor = sm._fd
    view = sm.buf[:]
    view[:] = b'12345678'
    sm.unlink()
    del sm
    gc.collect()
    assert errors == []
    assert bytes(view) == b'12345678'
    assert mapping() is not None
    if descriptor >= 0:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    view.release()
    gc.collect()
    assert mapping() is None
    assert errors == []


def test_proxy_shared_storage(proxy_factory, lookup):
    proxy = proxy_factory(lookup)
    payload = pickle.dumps(proxy)
    first = pickle.loads(payload)
    second = pickle.loads(payload)
    if isinstance(proxy, _reduction._MmapParentProxy):
        assert not first.call.values.flags.writeable
        with pytest.raises(ValueError, match='read-only'):
            first.call.values[3] = 100
    else:
        first.call.values[3] = 100
        assert second(3) == 130
    # Creating a proxy copies buffers; the original callable stays independent.
    assert lookup(3) == 33
    lookup.values[3] = 200
    assert second(3) == (
        33 if isinstance(proxy, _reduction._MmapParentProxy) else 130
    )


def test_proxy_in_worker(proxy_factory, lookup):
    proxy = proxy_factory(lookup)
    with ProcessPoolExecutor(max_workers=1) as executor:
        pid, value, writable = executor.submit(describe_lookup, proxy).result(
            timeout=30
        )
    assert pid != os.getpid()
    assert value == 33
    assert writable == isinstance(proxy, _reduction._ShmemParentProxy)


@pytest.mark.parametrize('chunksize', [1, 2, None])
@pytest.mark.parametrize('bound', [False, True])
def test_map_n_with_proxy(
    proxy_factory, proxy_mode, lookup, monkeypatch, chunksize, bound
):
    calls = []

    def move(call, mode=True):
        calls.append((call, mode))
        return proxy_factory(call)

    monkeypatch.setattr(_parallel, 'move_to_shmem', move)
    monkeypatch.setattr(_parallel, '_NUM_CPUS', 2)
    call = lookup.__call__ if bound else lookup
    # Own the executor so no workers survive test teardown.
    with ProcessPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(
            _parallel, 'get_executor', lambda *a, **kw: nullcontext(executor)
        )
        actual = list(
            _parallel.map_n(
                call,
                range(8),
                mp=proxy_mode,
                max_workers=2,
                chunksize=chunksize,
            )
        )
    assert actual == [i * 11 for i in range(8)]
    assert calls == [(call, proxy_mode)]


def test_default_proxy():
    with _reduction.move_to_shmem(partial(pow, 2)) as proxy:
        assert proxy(5) == 32
        assert pickle.loads(pickle.dumps(proxy))(5) == 32
        future = Future[int]()
        future.set_result(32)

        def submit():
            return future

        assert proxy.wrap_func(submit) is submit


def test_unknown_proxy_mode():
    with pytest.raises(ValueError, match='Unknown'):
        _reduction.move_to_shmem(abs, mode='unknown')


def test_proxy_defers_cleanup(proxy_factory, monkeypatch):
    closed = []
    proxy = proxy_factory(partial(pow, 2))
    monkeypatch.setattr(type(proxy), '_close', lambda _: closed.append(True))
    first, second = Future[None](), Future[None]()
    with proxy as active:
        assert active is proxy
        assert proxy.wrap_func(lambda: first)() is first
        assert proxy.wrap_func(lambda: second)() is second
    assert closed == []
    first.set_result(None)
    assert closed == []
    second.cancel()
    assert closed == [True]


def test_proxy_submit_failure(proxy_factory):
    proxy = proxy_factory(partial(pow, 2))

    def fail():
        raise ValueError('submit failed')

    with proxy, pytest.raises(ValueError, match='submit failed'):
        proxy.wrap_func(fail)()
    with pytest.raises(RuntimeError, match='closed'):
        proxy.wrap_func(lambda: Future())()


def test_proxy_exit_twice(proxy_factory):
    proxy = proxy_factory(partial(np.sum, np.arange(4)))
    with proxy:
        pass
    with proxy:
        pass


def test_proxy_completed_future(proxy_factory):
    proxy = proxy_factory(partial(pow, 2))
    future = Future()
    future.set_result(32)
    with proxy:
        assert proxy.wrap_func(lambda: future)().result() == 32
    with pytest.raises(RuntimeError, match='closed'):
        proxy.wrap_func(lambda: Future())()


def test_proxy_cleanup_claimed_before_close(proxy_factory, monkeypatch):
    proxy = proxy_factory(partial(pow, 2))
    started, release = Event(), Event()
    closed = []

    def close(_):
        closed.append(True)
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(type(proxy), '_close', close)
    future = Future()
    with proxy:
        proxy.wrap_func(lambda: future)()
    worker = Thread(target=future.set_result, args=(None,))
    worker.start()
    try:
        assert started.wait(timeout=5)
        with proxy:
            pass
        assert closed == [True]
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()


def exercise_lifetime_map(call, mode, scenario, references):
    match scenario:
        case 'source':
            with pytest.raises(ValueError, match='source failed'):
                _parallel.map_n(
                    call,
                    failing_source(),
                    mp=mode,
                    max_workers=1,
                    chunksize=1,
                    prefetch=4,
                )
            return None

        case 'short':
            results = _parallel.map_n(
                call, range(1), mp=mode, max_workers=1, chunksize=1, prefetch=4
            )
            gc.collect()
            assert references[0]() is not None
            assert list(results) == [0]
            return results

        case 'worker':
            results = _parallel.map_n(
                call, range(4), mp=mode, max_workers=1, chunksize=1, prefetch=4
            )
            gc.collect()
            assert references[0]() is not None
            with pytest.raises(ValueError, match='worker failed'):
                next(results)
            return results

        case 'close':
            results = _parallel.map_n(
                call, range(4), mp=mode, max_workers=1, chunksize=1, prefetch=4
            )
            gc.collect()
            assert references[0]() is not None
            assert next(results) == 0
            results.close()
            return results

        case _:
            raise NotImplementedError


@pytest.mark.parametrize('scenario', ['short', 'worker', 'source', 'close'])
def test_map_n_proxy_lifetime(proxy_mode, scenario, monkeypatch, tmp_path):
    monkeypatch.setattr(_reduction, '_SHM_ROOT', tmp_path)
    monkeypatch.setattr(_parallel, '_NUM_CPUS', 1)
    futures, references, resources, cleanups = [], [], [], []
    cleaned = Event()
    original_move = _parallel.move_to_shmem
    proxy_type = (
        _reduction._MmapParentProxy
        if proxy_mode == 'mmap'
        else _reduction._ShmemParentProxy
    )
    original_close = proxy_type._close

    def move(call, mode):
        proxy = original_move(call, mode)
        references.append(weakref.ref(proxy))
        if proxy_mode == 'shmem':
            resources.extend(sm.name for sm, _ in proxy.memos)
        return proxy

    def close(proxy):
        cleanups.append(all(f.done() for f in futures))
        original_close(proxy)
        cleaned.set()

    monkeypatch.setattr(_parallel, 'move_to_shmem', move)
    monkeypatch.setattr(proxy_type, '_close', close)
    with Manager() as manager, ProcessPoolExecutor(max_workers=1) as executor:
        gate = manager.Event()

        def submit(*args, **kwargs):
            future = executor.submit(*args, **kwargs)
            futures.append(future)
            return future

        monkeypatch.setattr(
            _parallel,
            'get_executor',
            lambda *a, **kw: nullcontext(SimpleNamespace(submit=submit)),
        )
        call = GatedLookup(gate, fail=scenario == 'worker')
        results = None
        try:
            results = exercise_lifetime_map(
                call, proxy_mode, scenario, references
            )
            if scenario != 'short':
                gc.collect()
                assert references[0]() is not None
                assert not cleaned.is_set()
        finally:
            gate.set()
            for index, future in enumerate(futures):
                if scenario == 'worker' and index == 0:
                    with pytest.raises(ValueError, match='worker failed'):
                        future.result(timeout=30)
                else:
                    assert future.result(timeout=30) == index
            assert cleaned.wait(timeout=5)
    assert cleanups == [True]
    del results
    futures.clear()
    del future
    gc.collect()
    assert not list(tmp_path.iterdir())
    for name in resources:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
