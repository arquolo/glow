__all__ = ['make_loader']

import functools
import os
from abc import abstractmethod
from typing import Any, Iterable, Protocol, Sequence, Tuple, TypeVar, Union

import torch
from torch.utils.data import Dataset, Sampler

from ..core import chunked, mapped, repeatable
from ..core.pipe.len_helpers import SizedIterable

_KT = TypeVar('_KT')
_KT_contra = TypeVar('_KT_contra', contravariant=True)
_NUM_CPUS: int = os.cpu_count()  # type: ignore


class _Mapping(Protocol[_KT_contra]):
    @abstractmethod
    def __getitem__(self, key: _KT_contra) -> Any:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


def _get_batch(dataset, indices):
    return tuple(
        torch.stack([torch.as_tensor(item) for item in row])
        for row in zip(*[dataset[index] for index in indices]))


def make_loader(
        dataset: Union[_Mapping[_KT], 'Dataset[Sequence]'],
        sampler: Union[Iterable[_KT], Sampler] = None,
        batch_size: int = 1,
        workers: int = _NUM_CPUS,
        multiprocessing: bool = True
) -> SizedIterable[Tuple[torch.Tensor, ...]]:
    """Yields batches of batch_size from dataset in order from sampler.

    Parameters:
    - batch_size - size of batch, each workers computes batch independently.
    - workers - Count of workers, by default all hardware threads are occupied.
    - multiprocessing - whether to use processes or threads.
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
            chunk_size=1 if multiprocessing else 0,
            workers=workers)

    return loop()
