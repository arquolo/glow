from __future__ import annotations

__all__ = [
    'Attention', 'BottleneckResidualBlock', 'DenseBlock', 'DenseDelta',
    'FeedForward', 'MbConv', 'MbConvPool', 'MobileNetV2Block',
    'MobileNetV2Pool', 'MobileNetV3Block', 'MobileNetV3Pool',
    'MultiAxisAttention', 'ResidualBlock', 'ResNeStBlock', 'ResNeStPool',
    'ResNeXtBlock', 'SplitAttention', 'SqueezeExcitation'
]

import torch
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torchvision.ops.stochastic_depth import StochasticDepth

from .aggregates import Cat, Ensemble, Gate, Residual
from .context import ConvCtx
from .util import ActivationFn, NameMixin

# --------------------------------- densenet ---------------------------------


def _round8(v: float, divisor: int = 8) -> int:
    """Ensure that number rounded to nearest 8, and error is less than 10%

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    n = v / divisor
    return int(max(n + 0.5, n * 0.9 + 1)) * divisor


class DenseBlock(nn.ModuleList):
    expansion = 4
    efficient = True
    __constants__ = ['step', 'depth', 'bottleneck']

    def __init__(self,
                 depth: int = 4,
                 step: int = 16,
                 bottleneck: bool = True,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        dim_inner = _round8(step * self.expansion)

        layers = []
        for _ in range(depth):
            layer: list[nn.Module] = []
            if bottleneck:
                layer += ctx.norm_act_conv(ctx.conv(dim_inner, 1))
            layer += ctx.norm_act_conv(ctx.conv(step, 3))
            layers.append(
                Cat(*layer) if self.efficient else nn
                .Sequential(Cat(), *layer))

        super().__init__(layers)

        self.step = step
        self.depth = depth
        self.bottleneck = bottleneck

    def __repr__(self) -> str:
        dim_in = next(m for m in self.modules()
                      if isinstance(m, nn.modules.conv._ConvNd)).in_channels

        dim_out = self.step * self.depth
        if not isinstance(self, DenseDelta):
            dim_out = (dim_in + dim_out) * (dim_in != 0)

        line = f'{dim_in}, {dim_out}, step={self.step}, depth={self.depth}'
        if self.bottleneck:
            line += ', bottleneck=True'
        return f'{type(self).__name__}({line})'

    def base_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = [x]
        for m in self:
            xs.append(m(xs))
        return xs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.base_forward(x)
        return torch.cat(xs, 1)


class DenseDelta(DenseBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.base_forward(x)
        return torch.cat(xs[1:], 1)  # Omit original x


# ---------------------------------- se-net ----------------------------------


class SqueezeExcitation(NameMixin, Gate):
    def __init__(self,
                 dim: int,
                 mlp_ratio: float = 0.25,
                 activation: ActivationFn = nn.SiLU,
                 scale_activation: ActivationFn = nn.Sigmoid):
        dim_inner = _round8(dim * mlp_ratio)
        super().__init__(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, dim_inner, bias=False),
            activation(),
            nn.Linear(dim_inner, dim, bias=False),
            scale_activation(),
            Rearrange('b c -> b c 1 1'),
        )
        self.name = f'{dim}, {dim_inner=}'


# ---------------------------------- resnet ----------------------------------


class ResidualBlock(nn.Sequential):
    """BasicBlock from ResNet-18/34"""
    def __init__(self,
                 dim: int,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.SiLU)
        super().__init__(
            Residual(
                ctx.conv(dim, 3, bias=False),
                ctx.norm(),
                ctx.activation(inplace=True),
                ctx.conv(dim, 3, bias=False),
                ctx.norm(),
                StochasticDepth(dropout, 'row'),
            ),
            ctx.activation(inplace=True),
        )


class BottleneckResidualBlock(nn.Sequential):
    """BottleneckBlock from ResNet-50/101/152"""

    # https://arxiv.org/abs/1512.03385
    def __init__(self,
                 dim: int,
                 bn_ratio: float = 0.25,
                 groups: int | None = 1,
                 se_ratio: float = 0.25,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.SiLU)
        dim_inner = _round8(dim * bn_ratio)
        super().__init__(
            Residual(
                ctx.conv(dim_inner, 1, bias=False),
                ctx.norm(),
                ctx.activation(inplace=True),
                ctx.conv(dim_inner, 3, groups=groups, bias=False),
                ctx.norm(),
                ctx.activation(inplace=True),
                ctx.conv(dim, 3, bias=False),
                ctx.norm(),
                *([SqueezeExcitation(dim, se_ratio, ctx.activation)]
                  if se_ratio else []),
                StochasticDepth(dropout, 'row'),
            ),
            ctx.activation(),
        )


class ResNeXtBlock(BottleneckResidualBlock):
    def __init__(self,
                 dim: int,
                 se_ratio: float = 0.25,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.SiLU)
        super().__init__(dim, 0.5, 32, se_ratio, dropout, ctx)


def _mobilenet_core(dim: int,
                    dim_inner: int,
                    dim_out: int,
                    stride: int = 1,
                    se: bool = False,
                    ctx: ConvCtx | None = None) -> list[nn.Module]:
    ctx = ctx or ConvCtx(activation=nn.ReLU)  # or nn.HardSwish

    children: list[nn.Module] = []
    if dim != dim_inner:
        children += [
            ctx.conv(dim_inner, 1, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True)
        ]
    children += [
        (ctx.conv(dim_inner, 3, groups=dim_inner, bias=False) if stride == 1
         else ctx.conv_pool(dim_inner, groups=dim_inner, bias=False)),
        ctx.norm(),
        ctx.activation(inplace=True),
    ]
    if se:
        children += [
            SqueezeExcitation(dim_inner, 0.25, ctx.activation, nn.Hardsigmoid)
        ]
    children += [
        ctx.conv(dim_out, 1, bias=False),
        ctx.norm(),
    ]
    return children


class MobileNetV2Block(Residual):
    def __init__(self,
                 dim: int,
                 bn_ratio: float = 6,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.ReLU6)
        dim_inner = _round8(dim * bn_ratio)
        children = _mobilenet_core(dim, dim_inner, dim, 1, False, ctx)
        super().__init__(*children, StochasticDepth(dropout, 'row'))


class MobileNetV2Pool(nn.Sequential):
    def __init__(self,
                 dim: int,
                 dim_out: int | None = None,
                 bn_ratio: float = 6,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.ReLU6)
        dim_out = dim_out or dim
        dim_inner = _round8(dim * bn_ratio)
        children = _mobilenet_core(dim, dim_inner, dim_out, 2, False, ctx)
        super().__init__(*children)


class MobileNetV3Block(Residual):
    def __init__(self,
                 dim: int,
                 dim_inner: int,
                 se: bool = False,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        children = _mobilenet_core(dim, dim_inner, dim, 1, se, ctx)
        super().__init__(*children, StochasticDepth(dropout, 'row'))


class MobileNetV3Pool(nn.Sequential):
    def __init__(self,
                 dim: int,
                 dim_inner: int,
                 dim_out: int | None = None,
                 se: bool = False,
                 ctx: ConvCtx | None = None):
        dim_out = dim_out or dim
        children = _mobilenet_core(dim, dim_inner, dim_out, 2, se, ctx)
        super().__init__(*children)


# ------------------------------- efficientnet -------------------------------


class MbConvPool(NameMixin, nn.Sequential):
    """
    Contrary from original article of MaxViT, uses post-norm MBConv

    x ← Proj(Pool(x)) + Proj(SE(DWConv(Conv(Norm(x)))))
    with:
        Norm = BatchNorm
        Conv = Conv1x1 + BatchNorm + GELU
        DWConv = Conv3x3-dw + BatchNorm + GELU
        SE = Squeeze-Excitation
        Proj = Conv1x1
    or, shortly:
     => seq(norm, conv1x1, norm, gelu, conv-dw-pool, norm, gelu, se, conv1x1)
      + seq(pool, conv1x1)
    """
    def __init__(self,
                 dim: int,
                 se_ratio: float = 0.25,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.SiLU, even=False)
        super().__init__(
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
            ctx.activation(True),
            ctx.conv_pool(dim, overlap=1, groups=dim, bias=False),
            ctx.norm(),
            ctx.activation(True),
            SqueezeExcitation(dim, se_ratio, ctx.activation),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),  # ! postnorm
        )
        self.name = f'{dim}, {se_ratio=}'


class MbConv(NameMixin, Residual):
    """
    Contrary from original article of MaxViT, uses post-norm MBConv

    x ← x + Proj(SE(DWConv(Conv(Norm(x)))))
    with:
        Norm = BatchNorm
        Conv = Conv1x1 + BatchNorm + GELU
        DWConv = Conv3x3-dw + BatchNorm + GELU
        SE = Squeeze-Excitation
        Proj = Conv1x1
    or, shortly:
     => x + seq(norm, conv1x1, norm, gelu, conv-dw, norm, gelu, se, conv1x1)
    """
    def __init__(self,
                 dim: int,
                 se_ratio: float = 0.25,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx(activation=nn.SiLU)
        super().__init__(
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
            ctx.activation(True),
            ctx.conv(dim, 3, groups=dim, bias=False),
            ctx.norm(),
            ctx.activation(True),
            SqueezeExcitation(dim, se_ratio, ctx.activation),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
            StochasticDepth(dropout, 'row'),
        )
        self.name = f'{dim}, {se_ratio=}, {dropout=}'


# --------------------------------- resnest ----------------------------------


class SplitAttention(NameMixin, nn.Module):
    """
    Split-Attention (aka Splat) block from ResNeSt.
    If radix == 1, equals to SqueezeExitation block from SENet.
    """
    __constants__ = NameMixin.__constants__ + ['radix']

    def __init__(self,
                 dim: int,
                 groups: int = 1,
                 radix: int = 2,
                 ctx: ConvCtx | None = None):
        assert dim % (groups * radix) == 0
        ctx = ctx or ConvCtx()
        dim_inner = dim * radix // 4

        super().__init__()
        self.radix = radix
        self.to_radix = Rearrange('b (r gc) h w -> b r gc h w', r=radix)
        self.attn = nn.Sequential(
            # Mean by radix and spatial dims
            Reduce('b r gc h w -> b gc 1 1', 'mean'),

            # Core
            ctx.conv(dim_inner, 1, groups=groups, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True),
            ctx.conv(dim * radix, 1, groups=groups, bias=False),

            # Normalize
            Rearrange('b (g r c) 1 1 -> b r (g c)', g=groups, r=radix),
            nn.Sigmoid() if radix == 1 else nn.Softmax(1),
        )
        self.name = f'{dim} -> {dim_inner} -> {dim}x{radix}, groups={groups}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rchw = self.to_radix(x)
        rc = self.attn(x * self.radix if self.radix > 1 else x)
        chw = torch.einsum('b r c h w, b r c -> b c h w', rchw, rc)
        return chw.contiguous()


class ResNeStBlock(nn.Sequential):
    def __init__(self,
                 dim: int,
                 radix: int = 1,
                 groups: int = 1,
                 rate: float = 0.25,
                 dropout: float = 0,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        dim_inner = _round8(dim * rate)

        core: list[nn.Module] = [
            ctx.conv(dim_inner, 1, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True),
            ctx.conv(dim_inner * radix, 3, groups=groups * radix, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True),
            SplitAttention(dim_inner, groups, radix),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
        ]
        super().__init__(
            Residual(*core, StochasticDepth(dropout, 'row')),
            nn.ReLU(inplace=True),
        )


class ResNeStPool(nn.Sequential):
    def __init__(self,
                 dim: int,
                 radix: int = 1,
                 groups: int = 1,
                 dropout: float = 0,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        dim_inner = (dim // 4) * groups
        core = [
            ctx.conv(dim_inner, 1, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True),
            ctx.conv(dim_inner * radix, 3, groups=groups * radix, bias=False),
            ctx.norm(),
            ctx.activation(inplace=True),
            SplitAttention(dim_inner, groups, radix),
            ctx.avg_pool(),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
        ]
        downsample = [
            ctx.avg_pool(),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
        ]
        super().__init__(
            Ensemble(
                [*core, StochasticDepth(dropout, 'row')],
                downsample,
                mode='sum',
            ),
            ctx.activation(inplace=True),
        )


# ------------------------------- transformers -------------------------------


class ReAttention(nn.Sequential):
    """
    Re-Attention from [DeepViT](https://arxiv.org/abs/2103.11886)
    """
    def __init__(self, heads: int) -> None:
        super().__init__(
            Rearrange('b h i j -> b i j h'),
            nn.Linear(heads, heads, bias=False),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j'),
        )
        nn.init.normal_(self[1].weight)


class Attention(NameMixin, nn.Module):
    """
    Multihead self-attention module (M-SA)
    from [ViT](https://openreview.net/pdf?id=YicbFdNTTy).

    Supports Re-attention mechanism
    from [DeepViT](https://arxiv.org/abs/2103.11886).
    """
    def __init__(self,
                 dim: int,
                 dim_head: int = 64,
                 dropout: float = 0.,
                 qkv_bias: bool = False,
                 reattention: bool = False):
        super().__init__()
        assert dim % dim_head == 0
        heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Sequential(
            fc := nn.Linear(dim, 3 * dim, bias=qkv_bias),
            Rearrange('b n (split h d) -> split b h n d', h=heads, split=3),
        )
        nn.init.normal_(fc.weight, 0, (2 / (dim + dim_head)) ** .5)

        self.attend = nn.Sequential(nn.Softmax(-1), nn.Dropout(dropout))
        if reattention:
            self.attend.append(ReAttention(heads))

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            fc := nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        nn.init.xavier_normal_(fc.weight)

        self.name = f'{dim}, {heads=}'
        if qkv_bias:
            self.name += ', qkv_bias=True'
        if dropout:
            self.name += f', {dropout=}'
        if reattention:
            self.name += ', reattention=True'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b n dim -> b h n d
        qkv = self.to_qkv(x)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # Compute weights for each token
        dots = torch.einsum('bhid,bhjd -> bhij', q, k)
        attn = self.attend(dots * self.scale)

        # Remix tokens using weights
        out = torch.einsum('bhij,bhjd -> bhid', attn, v)

        # Restore shape, b h n d -> b n (h d)
        return self.to_out(out)


class _RelativePositionalBias(nn.Module):
    def __init__(self, heads: int, window_size: int) -> None:
        super().__init__()

        wdiff = 2 * window_size - 1
        self.bias = nn.Sequential(
            nn.Embedding(wdiff ** 2, heads),
            Rearrange('i j h -> h i j'),
        )

        axis = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1)
        pos = (
            rearrange(grid, 'i j c -> (i j) 1 c') -
            rearrange(grid, 'i j c -> 1 (i j) c'))
        pos -= pos.min()
        indices = pos @ torch.tensor([wdiff, 1])
        self.register_buffer('indices', indices, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.bias(self.indices)


class MultiAxisAttention(NameMixin, nn.Module):
    """
    Multi-axis self-attention (Max-SA)
    from [MaxViT](https://arxiv.org/abs/2204.01697)
    """
    def __init__(self,
                 dim: int,
                 dim_head: int = 32,
                 dropout: float = 0.,
                 qkv_bias: bool = False,
                 window_size: int = 7):
        super().__init__()
        assert dim % dim_head == 0
        heads = dim // dim_head

        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            Rearrange('... i (s h d) -> s ... h i d', s=3, h=heads),
        )
        self.bias = _RelativePositionalBias(heads, window_size)

        self.attend = nn.Sequential(
            nn.Softmax(-1),
            nn.Dropout(dropout),
        )
        self.to_out = nn.Sequential(
            Rearrange('... h i d -> ... i (h d)'),
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.name = f'{dim}, {heads=}, {window_size=}'
        if qkv_bias:
            self.name += ', qkv_bias=True'
        if dropout:
            self.name += f', {dropout=}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... i d -> ... i d, self-attention over i
        q, k, v = self.to_qkv(x).unbind(0)  # ... h i d

        sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        sim += self.bias()
        attn = self.attend(sim)

        # aggregate
        out = torch.einsum('... i j, ... j d -> ... i d', attn, v)

        # combine heads out
        return self.to_out(out)


class FeedForward(NameMixin, nn.Sequential):
    def __init__(self, dim: int, ratio: float, dropout: float = 0.):
        dim_inner = _round8(dim * ratio)
        super().__init__(
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )
        self.name = f'{dim}, {dim_inner=}'
        if dropout:
            self.name += f', {dropout=}'
