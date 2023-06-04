from itertools import count, islice

import torch
from torch.utils import data

from glow.nn import get_loader


class Dataset(data.Dataset):
    def __getitem__(self, index):
        return index

    def __len__(self):
        return 5


class PassDataset(data.Dataset):
    def __getitem__(self, index):
        return index

    def __len__(self):
        return 1


class PassTensorDataset(data.Dataset):
    def __getitem__(self, index):
        return torch.as_tensor([index])

    def __len__(self):
        return 1


class Sampler(data.Sampler):
    def __init__(self, n):
        self.n = n
        self.count = count()

    def __iter__(self):
        return iter(islice(self.count, self.n))

    def __len__(self):
        return self.n


def test_loader():
    # No batch, no collate, no tensors
    loader = get_loader(Dataset())
    assert len(loader) == 5
    assert [*loader] == [0, 1, 2, 3, 4]

    # Batch -> tensors
    loader2 = loader.batch(1)
    assert len(loader2) == 5
    assert [x.tolist() for x in loader2] == [[0], [1], [2], [3], [4]]

    loader2 = loader.batch(2)
    assert len(loader2) == 3
    assert [x.tolist() for x in loader2] == [[0, 1], [2, 3], [4]]


def test_pass_loader():
    loader = get_loader(PassDataset()).shuffle(Sampler(5)).batch(1)
    assert len(loader) == 5
    assert [x.tolist() for x in loader] == [[0], [1], [2], [3], [4]]

    # 2nd epoch
    assert len(loader) == 5
    assert [x.tolist() for x in loader] == [[5], [6], [7], [8], [9]]


def test_tensor_loader():
    loader = get_loader(PassTensorDataset()).shuffle(Sampler(5)).batch(1)
    assert len(loader) == 5
    assert [x.tolist() for x in loader] == [[[0]], [[1]], [[2]], [[3]], [[4]]]

    assert len(loader) == 5
    assert [x.tolist() for x in loader] == [[[5]], [[6]], [[7]], [[8]], [[9]]]
