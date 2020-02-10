__all__ = ('Activation', 'View', 'Noise')

import functools

import torch
from torch import nn
from torch.nn import functional

from ..core import repr_as_obj


class Activation(nn.Module):
    closure = staticmethod(functional.relu)

    @classmethod
    def new(cls, inplace=True):
        module = cls()
        module.closure = functools.partial(cls.closure, inplace=inplace)
        return module

    def forward(self, x):
        return self.closure(x)

    def extra_repr(self):
        fn = self.closure.func
        return (f'fn={fn.__module__}.{fn.__qualname__},'
                f' {repr_as_obj(self.closure.keywords)}')


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self):
        return f'shape={(None, *self.shape)}'


class Noise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if not self.training:
            return x
        return torch.empty_like(x).normal_(std=self.std).add_(x)

    def extra_repr(self):
        return f'std={self.std}'
