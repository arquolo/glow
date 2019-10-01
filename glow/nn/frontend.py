__all__ = ('LazyLoader', )

from dataclasses import dataclass
from functools import partial
from typing import Iterator, Tuple

import torch
from torch.utils.data import (Dataset as _Dataset, Sampler as _Sampler)

from ..iters import mapped, chunked


def _to_tensor(x) -> torch.Tensor:
    return x if torch.is_tensor(x) else torch.tensor(x)


def _get_sample(dataset, index) -> Tuple[torch.Tensor]:
    return tuple(_to_tensor(item) for item in dataset[index])


@dataclass
class LazyLoader:
    dataset: _Dataset
    sampler: _Sampler
    batch_size: int = 1
    chunk_size: int = 0

    def __len__(self) -> int:
        batches, remainder = divmod(len(self.sampler), self.batch_size)
        return batches + bool(remainder)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor]]:
        samples = mapped(
            partial(_get_sample, self.dataset),
            self.sampler,
            offload=self.chunk_size or self.batch_size,
        )
        for batch in chunked(samples, self.batch_size):
            yield tuple(torch.stack(row) for row in zip(*batch))
