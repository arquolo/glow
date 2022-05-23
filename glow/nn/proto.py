from __future__ import annotations

__all__ = ['tiramisu']

from collections.abc import Iterable
from itertools import accumulate
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from .modules import (Attention, DenseBlock, DenseDelta, Ensemble, FeedForward,
                      MbConv, MbConvPool, MultiAxisAttention, PreNormResidual)
from .modules.context import ConvCtx
from .modules.util import pair


def tiramisu(num_classes: int,
             channels: int = 3,
             init: int = 48,
             depths: Iterable[int] = (4, 4),
             step: int = 12):
    """
    Implementation of
    [The One Hundred Layers Tiramisu](https://arxiv.org/pdf/1611.09326.pdf)

    See [FC-DenseNet](https://github.com/SimJeg/FC-DenseNet) as reference
    Theano/Lasagne-based implementation,
    and (https://github.com/bfortuner/pytorch_tiramisu) as pytorch fork.
    """
    *depths, = depths
    *dims, _ = accumulate([init] + [step * depth for depth in depths])

    core: list[nn.Module] = []

    ctx = ConvCtx()
    for depth_prev, depth in zip(reversed(depths[1:]), reversed(depths[:-1])):
        dim = dims.pop()
        core = [
            DenseBlock(depth_prev, step),
            Ensemble(
                nn.Identity(),
                [
                    ctx.norm(),
                    ctx.activation(inplace=True),
                    ctx.conv(dim, 1, bias=False),
                    ctx.max_pool(),
                    *core,
                    DenseDelta(depth, step),
                    # ? norm + act
                    ctx.conv_unpool(depth * step),
                ],
                mode='cat',
            ),
        ]

    core = [
        nn.Conv2d(channels, init, 3, padding=1),
        *core,
        DenseBlock(depths[0], step),
        # ? norm + act
        ctx.conv(num_classes, 1),
    ]
    return nn.Sequential(*core)


# ---------------------------- transformer blocks ----------------------------


class VitBlock(nn.Sequential):
    def __init__(self,
                 dim: int,
                 dim_head: int,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.,
                 qkv_bias: bool = False,
                 reattn: bool = False):
        super().__init__(
            PreNormResidual(
                Attention(dim, dim_head, dropout, qkv_bias, reattn)),
            PreNormResidual(FeedForward(dim, mlp_ratio, dropout)),
        )


class MaxVitBlock(nn.Sequential):
    def __init__(self,
                 dim: int,
                 dim_head: int,
                 window: int,
                 mlp_ratio: float,
                 dropout: float = 0.,
                 qkv_bias: bool = False):
        super().__init__(
            # block attention
            Rearrange('b d (x u) (y v) -> b x y (u v) d', u=window, v=window),
            PreNormResidual(
                MultiAxisAttention(dim, dim_head, dropout, qkv_bias, window)),
            PreNormResidual(FeedForward(dim, mlp_ratio, dropout)),
            Rearrange('b x y (u v) d -> b d (x u) (y v)', u=window, v=window),

            # grid attention
            Rearrange('b d (u x) (v y) -> b x y (u v) d', u=window, v=window),
            PreNormResidual(
                MultiAxisAttention(dim, dim_head, dropout, qkv_bias, window)),
            PreNormResidual(FeedForward(dim, mlp_ratio, dropout)),
            Rearrange('b x y (u v) d -> b d (u x) (v y)', u=window, v=window),
        )


# ------------------- position encoding for inductive bias -------------------


class PositionEncoding(nn.Module):
    def __init__(self, dim: int, image_size: int | tuple[int, ...],
                 patch_size: int | tuple[int, ...]):
        super().__init__()
        image_hw = pair(image_size)
        patch_hw = pair(patch_size)

        assert all(i % p == 0 for i, p in zip(image_hw, patch_hw))
        token_size = *(i // p for i, p in zip(image_hw, patch_hw)),

        self.position = nn.Parameter(torch.randn(1, dim, *token_size))

    def extra_repr(self) -> str:
        _, dim, *space = self.position.shape
        return f'features={dim}, size={space}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = self.position
        if x.shape[2:] != position.shape[2:]:
            position = F.interpolate(
                position, x.shape[2:], mode='bicubic', align_corners=False)
        x += self.position
        return x


# ------------------------------ token actions -------------------------------


class CatToken(nn.Module):  # [B, N, D] -> [B, 1 + N, D]
    def __init__(self, dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(dim))

    def extra_repr(self) -> str:
        return f'features={self.token.shape[-1]}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _ = x.shape
        tokens = repeat(self.token, 'd -> b 1 d', b=b)
        return torch.cat((tokens, x), dim=1)


class PopToken(nn.Module):  # [B, N, D] -> [B, D]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :]


# -------------------------------- models ------------------------------------


class ViT(nn.Sequential):
    def __init__(self,
                 *,
                 image_size: int | tuple[int, int],
                 patch_size: int | tuple[int, int],
                 num_classes: int,
                 dim: int,
                 depth: int,
                 dim_head: int = 64,
                 mlp_ratio: float = 4.,
                 pool: Literal['cls', 'mean'] = 'cls',
                 channels: int = 3,
                 dropout: float = 0.,
                 dropout_emb: float = 0.,
                 qkv_bias: bool = True):
        assert pool in {'cls', 'mean'}

        super().__init__()
        self.to_tokens = nn.Sequential(
            nn.Conv2d(channels, dim, patch_size, patch_size),
            PositionEncoding(dim, image_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.transformer = nn.Sequential(
            # Operates in (b n d) domain
            CatToken(dim),
            nn.Dropout(dropout_emb),
            *[
                VitBlock(dim, dim_head, mlp_ratio, dropout, qkv_bias)
                for _ in range(depth)
            ],
            Reduce('b n d -> b d', 'mean') if pool == 'mean' else PopToken(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )


class MaxViT(nn.Sequential):
    def __init__(self,
                 *,
                 num_classes: int,
                 dim: int,
                 depth: tuple[int, ...],
                 dim_head: int = 32,
                 dim_stem: int | None = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4,
                 se_ratio: float = 0.25,
                 qkv_bias: bool = False,
                 dropout: float = 0.1,
                 channels: int = 3):
        super().__init__()
        dim_stem = dim_stem or dim
        self.stem = nn.Sequential(
            nn.Conv2d(channels, dim_stem, 3, 2, padding=1),
            nn.Conv2d(dim_stem, dim_stem, 3, padding=1),
        )

        dims = *((2 ** i) * dim for i, _ in enumerate(depth)),
        dims = (dim_stem, *dims)

        layers = []
        for dim, depth_ in zip(dims[1:], depth):
            for idx in range(depth_):
                layers += [
                    MbConvPool(dim, se_ratio) if idx == 0 else MbConv(
                        dim, se_ratio, dropout),
                    MaxVitBlock(dim, dim_head, window_size, mlp_ratio, dropout,
                                qkv_bias),
                ]
        self.layers = nn.Sequential(*layers)

        self.head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )
