from __future__ import annotations

__all__ = [
    'as_seeded', 'SeededDataset', 'SeededSampler', 'SequentialSampler',
    'RandomSampler', 'SubsetRandomSampler', 'WeightedRandomSampler',
    'BatchSampler'
]

import threading
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import NamedTuple, TypeVar, final

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .. import chunked, sliced

T_co = TypeVar('T_co', covariant=True)

RNG = threading.local()


class Index(NamedTuple):
    item: object
    seed: np.random.SeedSequence


class SeededSampler(Sampler[Index]):
    """Base class for inheritance, all samplers must derive from it"""
    def __init__(self, seed: int | None = None):
        self.seed = np.random.SeedSequence(seed)

    def __len__(self) -> int:
        raise NotImplementedError

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        raise NotImplementedError

    @final
    def advance(self, epochs: int):
        self.seed.spawn(epochs)

    @final
    def __iter__(self) -> Iterator[Index]:
        epoch, = self.seed.spawn(1)
        return map(Index, self.sample(epoch), epoch.spawn(len(self)))


@final
class _SeededTorchSampler(SeededSampler):
    def __init__(self, source: Sampler, seed: int | None = None):
        super().__init__(seed)
        self.source = source

    def __len__(self) -> int:
        return len(self.source)  # type: ignore

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        torch.manual_seed(seed.generate_state(1, dtype='u8').item())
        return iter(self.source)


@final
class SeededDataset(Dataset[T_co]):
    def __init__(self, source: Dataset[T_co]):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)  # type: ignore

    def __getitem__(self, index: Index) -> T_co:
        seed = index.seed.generate_state(1, dtype='u8').item()
        torch.manual_seed(seed)

        RNG.current = np.random.default_rng(index.seed)
        return self.source[index.item]


def as_seeded(dataset: Dataset,
              sampler: Sampler,
              seed: int | None = None) -> tuple[Dataset, Sampler]:
    return SeededDataset(dataset), _SeededTorchSampler(sampler, seed)


# TODO: DistributedSampler without drop_last/pad_last (return all samples)


class SequentialSampler(SeededSampler):
    """Samples elements sequentially, always in the same order.

    Parameters:
    - source - dataset to sample from
    """
    def __init__(self, source: Sized, seed: int | None = None):
        super().__init__(seed=seed)
        self.source = source

    def sample(self, _: np.random.SeedSequence) -> Iterable:
        return range(len(self))

    def __len__(self) -> int:
        return len(self.source)


class RandomSampler(SeededSampler):
    """Samples elements randomly. If without replacement, then sample
    from a shuffled dataset.
    If with replacement, then user can specify `num_samples` to draw.

    Parameters:
    - source - dataset to sample from.
    - replacement - if set samples are drawn on-demand with replacement.
    - num_samples - number of samples to draw, default=`len(dataset)`.
      This argument is supposed to be specified only when `replacement` is set.
    - generator - Generator used in sampling.
    """
    def __init__(self,
                 source: Sized,
                 replacement: bool = False,
                 num_samples: int | None = None,
                 seed: int | None = None):
        super().__init__(seed=seed)
        self.source = source
        self.replacement = replacement
        self.num_samples = num_samples

        if num_samples is not None and not replacement:
            raise ValueError(
                'With replacement=False, num_samples should not be specified, '
                'since a random permute will be performed.')

        if not self:
            raise ValueError('num_samples should be a positive integer '
                             f'value, but got num_samples={len(self)}')

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        n = len(self.source)
        rng = np.random.default_rng(seed)

        if self.replacement:
            for s in sliced(range(len(self)), 32):
                yield from rng.integers(n, size=len(s)).tolist()
        else:
            yield from rng.permutation(n).tolist()

    def __len__(self) -> int:
        return self.num_samples or len(self.source)


class SubsetRandomSampler(SeededSampler):
    """Samples elements randomly from a given list of indices,
    without replacement.

    Parameters:
    - indices - a sequence of indices
    """
    def __init__(self, indices: Sequence[int], seed: int | None = None):
        super().__init__(seed=seed)
        self.indices = np.array(indices)

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        rng = np.random.default_rng(seed)
        return rng.permutation(self.indices).tolist()

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(SeededSampler):
    """Samples elements from ``[0,..,len(weights)-1]`` with given
    probabilities (weights).

    Parameters:
    - weights - a sequence of weights, not necessary summing up to one
    - num_samples - number of samples to draw
    - replacement - if set, samples are drawn with replacement.
      If not, they are drawn without replacement, which means that when a
      sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5,
        ...                            replacement=False))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5))
        [0, 1, 4, 3, 2]
    """
    def __init__(self,
                 weights: Sequence[float],
                 num_samples: int,
                 replacement: bool = True,
                 seed: int | None = None):
        if num_samples <= 0:
            raise ValueError('num_samples should be a positive integer '
                             f'value, but got num_samples={num_samples}')

        super().__init__(seed=seed)
        self.num_samples = num_samples
        self.replacement = replacement
        self.weights = np.array(weights, dtype='f8')
        self.weights /= self.weights.sum()

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        rng = np.random.default_rng(seed)
        return rng.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights).tolist()

    def __len__(self):
        return self.num_samples


class BatchSampler(SeededSampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Parameters:
    - sampler - Base sampler. Can be any iterable object
    - batch_size - Size of mini-batch.
    - drop_last - If set, the sampler will drop the last batch if
      its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(
        ...     SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(
        ...     SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self,
                 sampler: Sampler[int] | Iterable,
                 batch_size: int,
                 drop_last: bool,
                 seed: int | None = None):
        super().__init__(seed=seed)
        if batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def sample(self, seed: np.random.SeedSequence) -> Iterable:
        for batch in chunked(self.sampler, self.batch_size):
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        n = len(self.sampler)  # type: ignore
        if self.drop_last:
            return n // self.batch_size
        else:
            return len(range(n)[::self.batch_size])
