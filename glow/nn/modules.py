from __future__ import annotations

__all__ = [
    'Activation', 'Cat', 'Mish', 'Noise', 'Sum', 'Swish', 'UpsampleArea',
    'UpsamplePoint', 'View', 'resblock'
]

import functools
from collections.abc import Sequence
from typing import Literal

import torch
import torch.autograd
import torch.jit
import torch.nn.functional as F
from torch import nn

from .. import repr_as_obj


class Activation(nn.Module):  # TODO: deprecate and/or refactor
    closure = staticmethod(F.relu)

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


# ------------------------- EfficientNet activations -------------------------


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
            return F.softplus(x).tanh().mul(x)

        @staticmethod
        @torch.jit.script
        def _backward(x, grad):
            sig = x.sigmoid()
            tanh = F.softplus(x).tanh()
            return (tanh + x * sig * (1 - tanh * tanh)).mul(grad)


# ---------------------------- proper upsampling ----------------------------


class UpsampleArea(nn.Module):
    """Upsamples input image, treating samples as squares.

    Splits original samples to S interpolated ones, where S == `scale`.
    Result size is always multiple of `scale`.
    """

    _modes = {3: 'linear', 4: 'bilinear', 5: 'trilinear'}
    _offset = 0
    _alignment = False

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        size = tuple((s - self._offset) * self.scale + self._offset
                     for s in x.shape[2:])
        return F.interpolate(
            x, size, mode=self._modes[x.ndim], align_corners=self._alignment)

    def extra_repr(self):
        return f'scale={self.scale}'


class UpsamplePoint(UpsampleArea):
    """Upsamples input image, treating samples as points.

    Inserts S-1 interpolated samples between pair of original ones,
    where S is `scale`.
    Thus for scale `S`, and input size `N`, result size is `(N - 1) * S + 1`.
    """
    _offset = 1
    _alignment = True


# --------------------------------- joiners ---------------------------------


class Cat(nn.Sequential):
    def forward(self, x):
        return torch.cat([m(x) for m in self], dim=1)


class Sum(nn.Sequential):
    def forward(self, x):
        first, *rest = self
        return sum((m(x) for m in rest), first(x))


def cat_seq(*blocks):
    """Creates block returning `cat([seq(*blocks)(x), x])`, useful for U-Net"""
    return Cat(nn.Sequential(*blocks), nn.Identity())


# ---------------------------------- basics ----------------------------------


def _norm_fn(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


def _act_fn(inplace: bool = False) -> nn.Module:
    return nn.ReLU(inplace=inplace)


def conv(cin, cout=None, kernel=3, stride=1, padding=1):
    cout = cout or cin
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding=padding, bias=False),
        _norm_fn(cout),
        _act_fn(inplace=True),
    )


def upconv(cin: int, cout: int | None = None) -> nn.Sequential:
    cout = cout or cin
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1),
        _norm_fn(cout),
        _act_fn(inplace=True),
    )


# ---------------------------------- combos ----------------------------------


class _Named:
    name: str

    def __repr__(self):
        return self.name or super().__repr__()


class SEBlock(_Named, nn.Sequential):
    def __init__(self, cin: int, reduction: int = 16):
        mid = cin // reduction
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, cin, 1, bias=False),
            nn.Sigmoid(),
        )
        self.name = f'{type(self).__name__}({cin} -> {mid} -> {cin})'

    def forward(self, x):
        return x * super().forward(x)


class SplitAttention(_Named, nn.Sequential):
    def __init__(self, cin: int, groups: int = 1, radix: int = 2):
        mid = cin * radix // 4

        super().__init__(
            # Mean by radix and spatial dims
            View(groups, radix, cin // groups, -1),
            nn.AdaptiveAvgPool3d((1, None, 1)),  # type: ignore
            View(-1, 1, 1),
            # Core
            nn.Conv2d(cin, mid, 1, groups=groups, bias=False),
            _norm_fn(mid),
            _act_fn(inplace=True),
            nn.Conv2d(mid, cin * radix, 1, groups=groups, bias=False),
            # Normalize
            View(groups, radix, cin // groups),
            nn.Sigmoid() if radix == 1 else nn.Softmax(dim=2),
        )
        self.groups = groups
        self.radix = radix
        self.name = (f'{type(self).__name__}'
                     f'({cin} -> {mid} -> {cin}x{radix}, groups={groups}')

    def forward(self, x):
        b, _, h, w = x.shape
        return torch.einsum(
            'bgrc,bgrchw->bgchw',
            super().forward(x * self.radix),
            x.view(b, self.groups, self.radix, -1, h, w),
        ).view(b, -1, h, w)


def _resblock(
        core: Sequence[nn.Module],
        tail: Sequence[nn.Module] = (),
) -> nn.Sequential:
    return nn.Sequential(
        Sum(nn.Identity(), nn.Sequential(*core)),
        *tail,
    )


def _resblock_core(cin: int, bottleneck: bool, groups: int, radix: int,
                   expansion: int) -> list[nn.Module]:
    if not bottleneck:  # resnet-18/34
        return [
            nn.Conv2d(cin, cin, 3, padding=1, bias=False),
            _norm_fn(cin),
            _act_fn(inplace=True),
            nn.Conv2d(cin, cin, 3, padding=1, bias=False),
        ]

    if radix:  # resnest-50/...
        mid = round(cin * groups / expansion)
        sa = [SplitAttention(mid, groups=groups, radix=radix)]
        mid2 = mid * radix
        groups *= radix
    else:  # (wide)resnet/resnext-50/101/152
        mid = mid2 = round(cin / expansion)
        sa = []

    return [
        nn.Conv2d(cin, mid, 1, bias=False),
        _norm_fn(mid),
        _act_fn(inplace=True),
        nn.Conv2d(mid, mid2, 3, padding=1, groups=groups, bias=False),
        _norm_fn(mid2),
        _act_fn(inplace=True),
        *sa,
        nn.Conv2d(mid, cin, 1, bias=False),
    ]


def resblock(cin: int,
             se: bool = False,
             bottleneck: bool = False,
             groups: int = 1,
             radix: int = 1,
             expansion: int = 4,
             preact: Literal['no', 'base', 'full'] = 'no') -> nn.Sequential:
    """
    Modes:
    - basic: 3x3(cin, cin) -> 3x3(cin, cin)
    - bottleneck: 1x1(cin, mid) -> 3x3(mid, mid, groups) -> sa -> 1x1(mid, cin)
        where: mid = cout // expansion.

    For ResNet-18/34 use basic mode. Groups, expansion & radix are not used.
    For deeper nets:
    - groups=1:
        - ResNet-X: expansion=4
        - WideResNet: expansion=2
        - ResNeSt: expansion=4, radix=2 (with se = False)
    - groups=32:
        - ResNeXt-32x4d: expansion=2
        - ResNeXt-32x8d: expansion=1

    If preactivation is used, first resblock should use only 'base' mode,
    subsequent blocks should use 'full'.
    """
    assert not (se and radix)

    core = _resblock_core(cin, bottleneck, groups, radix, expansion)
    norm = _norm_fn(cin)
    act = _act_fn(inplace=True)
    se_block = [SEBlock(cin)] if se else ()

    if preact == 'full':  # fully pre-activated block
        return _resblock([norm, act, *core, *se_block])
    elif preact == 'base':
        return _resblock([*core, *se_block])  # first pre-activated block
    else:
        return _resblock([*core, norm, *se_block], [act])
