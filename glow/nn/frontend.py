__all__ = ('make_loader', )

from functools import partial
from typing import Iterable, Mapping, Tuple, TypeVar

import torch

from ..iters import make_sized, mapped, chunked

KT = TypeVar('KT')


def _get_sample(dataset, index):
    return tuple(torch.as_tensor(item) for item in dataset[index])


def _collate_fn(batch):
    return tuple(torch.stack(row) for row in zip(*batch))


def size_hint(dataset, sampler=None, batch_size=1, **_):
    if sampler is None:
        sampler = range(len(dataset))
    return len(range(0, len(sampler), batch_size))


@make_sized(hint=size_hint)
def make_loader(dataset: Mapping[KT, Tuple],
                sampler: Iterable[KT] = None,
                batch_size: int = 1,
                chunk_size: int = None,
                workers: int = None) -> Iterable[Tuple[torch.Tensor]]:
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
        sampler = range(len(dataset))
    if chunk_size is None:
        chunk_size = batch_size

    samples = mapped(
        partial(_get_sample, dataset),
        sampler,
        offload=chunk_size,
        workers=workers,
    )
    return mapped(_collate_fn, chunked(samples, batch_size), workers=0)
