import torch
from torch.utils.data import Dataset, Sampler

# TODO: make item seed invariant from num_workers
# TODO   for Map-Style Dataset (draft below) and IterableDataset


class _DeterministicDataset(Dataset):
    def __init__(self, source: Dataset):
        self.source = source

    def __getitem__(self, big_index):
        seed, index = big_index
        torch.manual_seed(seed)
        return self.source[index]

    def __len__(self):
        return len(self.source)  # type: ignore


class _DeterministicSampler(Sampler):
    def __init__(self, source: Sampler):
        self.source = source
        self.epoch = 0

    def __iter__(self):
        seeds = torch.randint(2 ** 32, size=[len(self.source)])  # type: ignore
        return zip(seeds, self.source)

    def __len__(self):
        return len(self.source)  # type: ignore

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def as_deterministic(dataset, sampler):
    return _DeterministicDataset(dataset), _DeterministicSampler(sampler)
