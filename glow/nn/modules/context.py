from __future__ import annotations

__all__ = ['ConvCtx']

import warnings
from dataclasses import dataclass, replace
from typing import Literal

from torch import nn

from .util import ActivationFn, LazyConvFn, LazyNormFn

warnings.filterwarnings('ignore', module='torch.nn.modules.lazy')


@dataclass(frozen=True)
class ConvCtx:
    """
    Mode:
    - even:
        kernel = 2 * overlap + stride
    - odd:
        kernel = 2 * overlap + 1

    Padding:
    - same padding = overlap
    - valid padding = 0
    """
    conv_fn: LazyConvFn = nn.LazyConv2d
    norm: LazyNormFn = nn.LazyBatchNorm2d
    activation: ActivationFn = nn.ReLU
    even: bool = False
    padding: Literal['same', 'valid'] = 'same'

    def _get_k(self, stride: int, overlap: int) -> int:
        return (stride if self.even else 1) + 2 * overlap

    def _get_p(self, kernel: int, stride: int, dilation: int) -> int:
        assert stride == 1 or dilation == 1, \
            'one of stride/dilation should be always 1'
        if stride == 1 and kernel % 2 == 0 and dilation % 2 != 0:
            raise ValueError('Even kernel with odd dilation is not supported')

        if self.even:
            total_padding = kernel - stride
            assert total_padding >= 0, \
                'kernel should be same or greater than stride'
            assert total_padding >= 0
        else:
            total_padding = kernel - 1

        assert total_padding % 2 == 0, \
            'padding is not symmetric, offset kernel by 1'
        if self.padding == 'valid':
            return 0
        return (total_padding // 2) * dilation

    def _invert(self) -> ConvCtx:
        return replace(self, even=not self.even)

    def conv(self,
             dim: int,
             kernel: int = 3,
             dilation: int = 1,
             groups: int | None = 1,
             bias: bool = True) -> nn.modules.conv._ConvNd:
        groups = groups or dim
        padding = self._get_p(kernel, 1, dilation)
        return self.conv_fn(dim, kernel, 1, padding, dilation, groups, bias)

    def conv_pool(self,
                  dim: int,
                  stride: int = 2,
                  overlap: int = 0,
                  groups: int | None = 1,
                  bias: bool = True) -> nn.modules.conv._ConvNd:
        groups = groups or dim
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return self.conv_fn(dim, kernel, stride, padding, 1, groups, bias)

    def avg_pool(self, stride: int = 2, overlap: int = 0) -> nn.AvgPool2d:
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return nn.AvgPool2d(kernel, stride, padding)

    def max_pool(self, stride: int = 2, overlap: int = 0) -> nn.MaxPool2d:
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return nn.MaxPool2d(kernel, stride, padding)

    def conv_unpool(self,
                    dim: int,
                    stride: int = 2,
                    overlap: int = 0,
                    groups: int | None = 1,
                    bias: bool = True) -> nn.ConvTranspose2d:
        groups = groups or dim
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return nn.LazyConvTranspose2d(dim, kernel, stride, padding, 0, groups,
                                      bias)

    def conv_norm(self, mod: nn.modules.conv._ConvNd) -> nn.Sequential:
        assert self.norm
        mod.bias = None
        return nn.Sequential(mod, self.norm())

    def conv_norm_act(self,
                      mod: nn.modules.conv._ConvNd,
                      inplace: bool = True) -> nn.Sequential:
        norms: list[nn.Module] = []
        if self.norm:
            mod.bias = None
            norms += [self.norm()]
        return nn.Sequential(mod, *norms, self.activation(inplace=inplace))

    def norm_act_conv(self,
                      mod: nn.modules.conv._ConvNd,
                      inplace: bool = True) -> nn.Sequential:
        assert self.norm
        mod.bias = None
        return nn.Sequential(self.norm(), self.activation(inplace=inplace),
                             mod)
