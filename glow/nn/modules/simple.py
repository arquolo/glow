__all__ = ['Bias2d', 'Conv2dWs', 'Noise', 'Upscale2d']

import torch
from torch import nn

from .. import functional as F


class Noise(nn.Module):
    __constants__ = ['std']

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return torch.empty_like(x).normal_(std=self.std).add_(x)

    def extra_repr(self) -> str:
        return f'std={self.std}'


class Upscale2d(nn.Module):
    """Upsamples input tensor in `scale` times.
    Use as inverse for `nn.Conv2d(kernel=3, stride=2)`.

    There're 2 different methods:

    - Pixels are thought as squares. Aligns the outer edges of the outermost
      pixels.
      Used in `torch.nn.Upsample(align_corners=True)`.

    - Pixels are thought as points. Aligns centers of the outermost pixels.
      Avoids the need to extrapolate sample values that are outside of any of
      the existing samples.
      In this mode doubling number of pixels doesn't exactly double size of the
      objects in the image.

    This module implements the second way (match centers).
    New image size will be computed as follows:
        `destination size = (source size - 1) * scale + 1`

    For comparison see [here](http://entropymine.com/imageworsener/matching).
    """
    __constants__ = ['stride']

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.upscale2d(x, self.stride)

    def extra_repr(self):
        return f'stride={self.stride}'


class Conv2dWs(nn.Conv2d):
    """
    [Weight standartization](https://arxiv.org/pdf/1903.10520.pdf).
    Better use with GroupNorm(32, features).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d_ws(x, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
