from __future__ import annotations

__all__ = ['conv2d_ws', 'upscale2d']

from typing import Union

import torch
import torch.nn.functional as F

_size = Union[torch.Size, list[int], tuple[int, ...]]


# @torch.jit.script
def upscale2d(x: torch.Tensor, stride: int = 2) -> torch.Tensor:
    # ! stride-aware fallback, works everywhere
    # x = F.interpolate(x, None, self.stride)
    # return F.avg_pool2d(x, self.stride, 1, 0)

    # ! matches to single libtorch op, complex torchscript op
    pad = 1 - stride
    size = [x.shape[2], x.shape[3]]
    size = [s * stride + pad for s in size]
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


def conv2d_ws(x: torch.Tensor,
              weight: torch.Tensor,
              bias: torch.Tensor | None = None,
              stride: _size | int = 1,
              padding: _size | int | str = 0,
              dilation: _size | int = 1,
              groups: int = 1):
    weight = F.layer_norm(weight, weight.shape[1:], eps=1e-5)
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
