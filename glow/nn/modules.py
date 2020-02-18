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


class _ModuleBase(nn.Module):
    class _AutoFn(torch.autograd.Function):
        @classmethod
        def forward(cls, ctx, x):
            ctx.save_for_backward(x)
            return cls._forward(x)

        @classmethod
        def backward(cls, ctx, grad):
            return cls._backward(ctx.saved_tensors[0], grad)

    def forward(self, x):
        return self._AutoFn.apply(x)


class Swish(_ModuleBase):
    class _AutoFn(_ModuleBase._AutoFn):
        @staticmethod
        @torch.jit.script
        def _forward(x):
            return x.sigmoid().mul(x)

        @staticmethod
        @torch.jit.script
        def _backward(x, grad):
            sig = x.sigmoid()
            return (sig * (1 + x * (1 - sig))).mul(grad)


class Mish(_ModuleBase):
    class _AutoFn(_ModuleBase._AutoFn):
        @staticmethod
        @torch.jit.script
        def _forward(x):
            return functional.softplus(x).tanh().mul(x)

        @staticmethod
        @torch.jit.script
        def _backward(x, grad):
            sig = x.sigmoid()
            tanh = functional.softplus(x).tanh()
            return (tanh + x * sig * (1 - tanh * tanh)).mul(grad)
