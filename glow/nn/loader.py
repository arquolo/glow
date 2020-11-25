__all__ = ['make_loader']

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from ..core import chunked, mapped

_CollateFn = Callable[[Any], Tuple[torch.Tensor, ...]]
_NUM_CPUS: int = os.cpu_count() or 1


def _default_collate(batch) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.stack([torch.as_tensor(item) for item in row])
        for row in zip(*batch))


@dataclass
class _Loader:
    dataset: Dataset
    batch_size: int
    sampler: Sampler
    num_workers: int
    mp: bool
    collate_fn: Optional[_CollateFn]
    chunksize: Optional[int]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        samples = mapped(
            self.dataset.__getitem__,
            self.sampler,
            mp=self.mp,
            chunksize=self.chunksize,
            num_workers=self.num_workers)
        batches = chunked(samples, self.batch_size)
        return (batches if self.collate_fn is None  # type: ignore
                else map(self.collate_fn, batches))

    def __len__(self) -> int:
        indices = range(0, len(self.sampler), self.batch_size)  # type: ignore
        return len(indices)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler = None,
    num_workers: int = _NUM_CPUS,
    multiprocessing: bool = True,
    collate_fn: Optional[_CollateFn] = _default_collate,
    chunk_from_batch: bool = False,
) -> _Loader:
    """Yields batches of batch_size from dataset in order from sampler.

    Parameters:
    - batch_size - size of batch, each workers computes batch independently.
    - workers - Count of workers, by default all hardware threads are occupied.
    - multiprocessing - whether to use processes or threads.
    - chunk_from_batch - Set count of samples to pass each worker. If set
      then chunksize will be equal to batch_size, otherwise it will be
      estimated automatically.
    """
    if sampler is None:
        sampler = range(len(dataset))  # type: ignore
    assert sampler is not None

    chunksize = batch_size if multiprocessing and chunk_from_batch else None

    return _Loader(dataset, batch_size, sampler, num_workers, multiprocessing,
                   collate_fn, chunksize)
