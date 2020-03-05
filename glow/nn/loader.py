__all__ = ('make_loader', )

import os
import functools
from typing import Iterable, Mapping, Sequence, TypeVar

import torch

from ..core import chunked, repeatable, mapped
from ..core.pipe.len_helpers import SizedIterable

_KT = TypeVar('_KT')
_NUM_CPUS: int = os.cpu_count()  # type: ignore


def _get_batch(dataset, indices):
    return tuple(
        torch.stack([torch.as_tensor(item) for item in row])
        for row in zip(*[dataset[index] for index in indices]))


def make_loader(dataset: Mapping[_KT, Sequence],
                sampler: Iterable[_KT] = None,
                batch_size=1,
                workers=_NUM_CPUS,
                multiprocessing=True) -> SizedIterable[Sequence[torch.Tensor]]:
    """Yields batches of `batch_size` from `dataset` in order from  `sampler`.

    Parameters:
      - `batch_size` - size of batch, each workers computes batch independently
        (default: `1`)
      - `workers` - count of worker threads/processes
        (default: same as `os.cpu_count()`)
      - `multiprocessing` - whether to use ProcessPool or ThreadPool
        (default: `True`)
    """
    if sampler is None:
        sampler = range(len(dataset))  # type: ignore

    assert sampler is not None
    size = len(range(0, len(sampler), batch_size))  # type: ignore
    chunked_getter = functools.partial(_get_batch, dataset)

    @repeatable(hint=lambda: size)
    def loop():
        chunked_sampler = chunked(sampler, batch_size)
        return mapped(
            chunked_getter,
            chunked_sampler,
            chunk_size=multiprocessing,
            workers=workers)

    return loop()
