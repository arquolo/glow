__all__ = ('Loader', )

from dataclasses import dataclass
from functools import partial
from typing import Generic, Iterable, Iterator, Mapping, Tuple, TypeVar

import torch

from ..iters import mapped, chunked

KT = TypeVar('KT')


def _get_sample(dataset, index):
    return tuple(torch.as_tensor(item) for item in dataset[index])


@dataclass
class Loader(Generic[KT]):
    dataset: Mapping[KT, Tuple]
    sampler: Iterable[KT] = None
    batch_size: int = 1
    chunk_size: int = None
    workers: int = None

    def __post_init__(self):
        if self.sampler is None:
            self.sampler = range(len(self.dataset))
        if self.chunk_size is None:
            self.chunk_size = self.batch_size

    def __len__(self) -> int:
        batches, remainder = divmod(len(self.sampler), self.batch_size)
        return batches + bool(remainder)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor]]:
        samples = mapped(
            partial(_get_sample, self.dataset),
            self.sampler,
            offload=self.chunk_size,
            workers=self.workers,
        )
        for batch in chunked(samples, self.batch_size):
            yield tuple(torch.stack(row) for row in zip(*batch))
