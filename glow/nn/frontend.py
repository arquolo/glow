__all__ = ('make_loader', )

import os
import functools
from typing import Iterable, Mapping, Optional, Sequence, TypeVar

import torch

from ..iters import chunked, repeatable, mapped

_KT = TypeVar('_KT')
_NUM_CPUS: int = os.cpu_count()  # type: ignore


def _get_sample(dataset, index):
    return tuple(torch.as_tensor(item) for item in dataset[index])


def _collate_fn(batch):
    return tuple(torch.stack(row) for row in zip(*batch))


def make_loader(dataset: Mapping[_KT, Sequence],
                sampler: Optional[Iterable[_KT]] = None,
                batch_size: int = 1,
                chunk_size: Optional[int] = None,
                workers: int = _NUM_CPUS) -> Iterable[Sequence[torch.Tensor]]:
    """Yields batches of `batch_size` from `dataset` in order from  `sampler`.

    Parameters:
      - `batch_size` - size of batch
        (default: `1`)
      - `chunk_size` - size of chunk to pass for each worker
        If `0`, threads are used
        (default: same as `batch_size`)
      - `workers` - count of workers
        (default: same as `os.cpu_count()`)
    """
    if sampler is None:
        sampler = range(len(dataset))  # type: ignore
    if chunk_size is None:
        chunk_size = batch_size

    assert sampler is not None
    size = len(range(0, len(sampler), batch_size))  # type: ignore
    getter = functools.partial(_get_sample, dataset)

    @repeatable(hint=lambda: size)
    def loop():
        samples = mapped(
            getter,
            sampler,
            chunk_size=chunk_size,
            workers=workers,
        )
        return mapped(_collate_fn, chunked(samples, batch_size), workers=0)

    return loop()
