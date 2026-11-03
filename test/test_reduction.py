import gc
import os
import pickle
from concurrent.futures import Future
from contextlib import ExitStack, nullcontext
from functools import partial
from multiprocessing.shared_memory import SharedMemory

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
    with pytest.raises(RuntimeError, match='closing'):
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
    with pytest.raises(RuntimeError, match='closing'):
        proxy.wrap_func(lambda: Future())()
