from __future__ import annotations

__all__ = ['get_loader']

import os
import re
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import (Dataset, IterableDataset, RandomSampler, Sampler,
                              SequentialSampler)
from torch.utils.data._utils import worker as torch_worker

from .. import buffered, chunked, map_n, roundrobin
from ..core._parallel import _get_executor, max_cpu_count
from ..distributed import get_ddp_info
from ._sampler import DdpSampler, SamplerLike, generate_seed
from .util import _apply

_NUM_CPUS: int = os.cpu_count() or 1

# ------------------------------- common bases -------------------------------


class _Loader(Protocol):
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


# ------------------------------ memory pinning ------------------------------


def pin_memory(data):
    def _pin_memory(x):
        return x.pin_memory()

    return _apply(data, _pin_memory)


@dataclass(frozen=True)
class _PinningLoader(_Loader):
    base: _Loader

    def __iter__(self) -> Iterator:
        return map_n(pin_memory, self.base, max_workers=1)

    def __len__(self) -> int:
        return len(self.base)


class _PinnableLoader(_Loader):
    def pin(self) -> _Loader:
        """
        Copies Tensors into device/CUDA pinned memory before returning them.
        Works in extra thread.
        """
        return _PinningLoader(self) if torch.cuda.is_available() else self


# --------------------------------- batching ---------------------------------


class _CollateFn(Protocol):
    def __call__(self, __items: tuple) -> Any:
        ...


@dataclass(frozen=True)
class _BatchedLoader(_PinnableLoader):
    base: _Loader
    batch_size: int
    collate_fn: _CollateFn
    drop_last: bool
    workers: int

    def __iter__(self) -> Iterator:
        it = iter(self.base)

        if self.drop_last:
            total = len(self) * self.batch_size
            it = islice(it, total)

        batches = chunked(it, self.batch_size)
        return map_n(self.collate_fn, batches, max_workers=self.workers)

    def __len__(self) -> int:
        len_ = len(self.base)

        if self.drop_last:
            return len_ // self.batch_size

        return len(range(0, len_, self.batch_size))


class _BatchableLoader(_PinnableLoader):
    def batch(self,
              batch_size: int,
              collate_fn: _CollateFn | None = None,
              drop_last: bool = False,
              daemon: bool = False) -> _BatchedLoader:
        """
        Groups data into batches.

        Parameters:
        - batch_size - How many samples per batch to load
        - collate_fn - Merges a list of samples to form a mini-batch of
          Tensor(s).
        - drop_last - Set to `True` to drop the last incomplete batch,
          if size of underlying loader is not divisible by the batch size.
          Otherwise the last batch can be smaller than others.
        - daemon - Set to do batching in background thread.
        """
        if collate_fn is None:
            collate_fn = collate
        # TODO: return _BatchedLoader(
        #   replace(self, finalize=None),
        #   batch_size, collate_fn, drop_last,
        # )
        return _BatchedLoader(self, batch_size, collate_fn, drop_last,
                              int(daemon))

    def shuffle(self,
                sampler: Sampler | SamplerLike | None) -> _BatchableLoader:
        """
        Reshuffle data at every epoch.

        Parameters:
        - sampler - Defines the strategy to draw samples from the dataset.
          Can be any `Iterable` with `__len__` implemented.
        """
        raise NotImplementedError


# ---------------------- loader for map-style datasets ----------------------


@dataclass(frozen=True)
class _MapLoader(_BatchableLoader):
    dataset: Dataset
    sampler: Sampler

    def __iter__(self) -> Iterator:
        return map(self.dataset.__getitem__, self.sampler)

    def __len__(self) -> int:
        assert isinstance(self.sampler, Sized)
        return len(self.sampler)

    def shuffle(self,
                sampler: Sampler | SamplerLike | bool | None) -> _MapLoader:
        if not sampler:
            return self

        if sampler is True:
            assert isinstance(self.dataset, Sized)
            sampler = RandomSampler(self.dataset)

        return replace(self, sampler=DdpSampler(sampler))


@dataclass(frozen=True)
class _MapMultiLoader(_MapLoader):
    max_workers: int
    mp: bool
    chunksize: int | None = None

    def __iter__(self) -> Iterator:
        max_workers = self.max_workers
        if ddp := get_ddp_info():
            max_workers //= ddp.world

        return map_n(
            self.dataset.__getitem__,
            self.sampler,
            max_workers=max_workers,
            chunksize=self.chunksize,
            mp=self.mp)


# -------------------- loader for iterable-style datasets --------------------


@dataclass(frozen=True)
class _Worker:
    dataset: IterableDataset
    id: int
    num_workers: int
    seed: int | None = None

    def __iter__(self) -> Iterator:
        torch_worker._worker_info = self  # type: ignore[assignment]
        try:
            yield from self.dataset
        finally:
            torch_worker._worker_info = None


@dataclass(frozen=True)
class _IterableLoader(_BatchableLoader):
    dataset: IterableDataset

    def __iter__(self) -> Iterator:
        return buffered(self.dataset)

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)
        return len(self.dataset)


@dataclass(frozen=True)
class _IterableMultiLoader(_IterableLoader):
    max_workers: int

    def __iter__(self) -> Iterator:
        seed = generate_seed()
        workers = [
            _Worker(self.dataset, idx, self.max_workers, seed)
            for idx in range(self.max_workers)
        ]
        with _get_executor(self.max_workers, True) as ex:
            yield from roundrobin(*(buffered(w, mp=ex) for w in workers))


# ----------------------------- factory function -----------------------------


def get_loader(dataset: Dataset,
               max_workers: int | None = 0,
               mp: bool = False) -> _BatchableLoader:
    """
    Data loader. Combines a dataset and a sampler (via shuffle method), and
    provides an iterable over the given dataset.

    The data loader supports both map-style and iterable-style datasets with
    single- or multi-process loading, customizing loading order and
    automatic batching (collation) and memory pinning.

    Differences from torch.utils.data.DataLoader:
    - Support of threadpool backend for map-style datasets.
    - Automatically adjusts chunksize of data for passing to the workers to
      reduce IPC overhead when multiple processes are used.
    - Automatically adjusts number of workers in distributed context for
      map-style datasets.

    Parameters:
    - dataset - Dataset from which to load the data.
    - max_workers - How many threads or subprocesses to use for data loading.
      `0` means that the data will be loaded in the main process/thread.
      `None` means all logical processors.
    - mp - Whether to use multiprocessing or not.
    """
    if max_workers is None:
        max_workers = max_cpu_count(_NUM_CPUS, mp)

    if isinstance(dataset, IterableDataset):
        if not mp and max_workers != 0:
            warnings.warn(
                'For iterable-style datasets multithreading is not supported. '
                'Setting max_workers to 0',
                stacklevel=2)
            max_workers = 0

        if (ddp := get_ddp_info()) and ddp.world > 1:
            raise ValueError(
                'For iterable-style datasets distributed use is not '
                'supported')

        if not max_workers:
            return _IterableLoader(dataset)
        return _IterableMultiLoader(dataset, max_workers)

    else:  # noqa; RET505
        if not isinstance(dataset, Sized):
            raise TypeError("dataset should be sized when it's not iterable")

        sampler = SequentialSampler(dataset)

        if not max_workers:
            return _MapLoader(dataset, DdpSampler(sampler))
        return _MapMultiLoader(dataset, DdpSampler(sampler), max_workers, mp)


# ---------------- convert & collate. forked from pytorch 2.0 ----------------

# TODO: split `collate` to `convert` (within worker) + `collate` (in main)
# TODO: i.e. "Loader".tensors [alters workers] .batch() [adds collation]


def convert(x):  # noqa: PLR0911
    tp = type(x)
    if isinstance(x, torch.Tensor):
        return x

    if tp.__module__ == 'numpy' and not isinstance(x, np.str_ | np.bytes_):
        if (isinstance(x, np.ndarray)
                and _NP_STR_DTYPE_PATTERN.search(x.dtype.str)):
            return x
        return torch.as_tensor(x)

    if isinstance(x, Mapping):
        return _apply_type(tp, {k: convert(v) for k, v in x.items()})

    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        list_ = [convert(xx) for xx in x]

        if isinstance(x, tuple) and hasattr(x, '_fields'):  # namedtuple
            return tp(*list_)
        return _apply_type(tp, list_)

    return x


def collate(batch):
    x0 = batch[0]
    tp = type(x0)

    if _HINT is not None:  # Fast alternative to functools.singledispatch
        fn = _HINT.get(tp) or next(
            (fn for tp_, fn in _HINT.items() if isinstance(x0, tp_)), None)
        if fn is not None:
            return fn(batch)

    if isinstance(x0, Mapping):
        return _apply_type(tp, {k: collate([x[k] for x in batch]) for k in x0})

    if isinstance(x0, Sequence):
        assert len({len(x) for x in batch}) <= 1  # py3.10+: zip(strict=True)
        list_ = [collate(samples) for samples in zip(*batch)]

        if isinstance(x0, tuple) and hasattr(x0, '_fields'):  # namedtuple
            return tp(*list_)
        return _apply_type(tp, list_)

    raise TypeError(_COLLATE_ERROR_MSG.format(tp))


def _apply_type(tp, x):
    try:
        return tp(x)
    except TypeError:
        return x


def _collate_tensor(batch: Sequence[torch.Tensor]):
    x = batch[0]
    out = None
    if torch_worker.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = x._typed_storage()._new_shared(numel, device=x.device)
        out = x.new(storage).resize_(len(batch), *x.shape)

    return torch.stack([*batch], out=out)


def _collate_ndarray(batch: Sequence[np.ndarray]):
    x = batch[0]
    if _NP_STR_DTYPE_PATTERN.search(x.dtype.str):
        raise TypeError(_COLLATE_ERROR_MSG.format(x.dtype))

    return collate([torch.as_tensor(x) for x in batch])


def _nop(batch):
    return batch


_HINT: dict[type | tuple[type, ...], Callable] = {
    torch.Tensor: _collate_tensor,
    np.ndarray: _collate_ndarray,  # For both ndarray and memmap
    # Skip strings
    bytes: _nop,
    str: _nop,
    # Tensorify scalars
    (np.bool_, np.number, np.object_): torch.as_tensor,  # py3.10: UnionType
    float: partial(torch.tensor, dtype=torch.float64),
    int: torch.tensor,
}

_NP_STR_DTYPE_PATTERN = re.compile('[SOUa]')
_COLLATE_ERROR_MSG = (
    'default_collate: batch must contain tensors, numpy arrays, numbers, '
    'dicts or lists; found {}')
