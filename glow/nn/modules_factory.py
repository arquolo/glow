from __future__ import annotations

__all__ = ['linear', 'conv', 'Cat', 'Sum', 'DenseBlock', 'SEBlock']

import random

import torch
from torch import nn

from .modules import Activation


class Nonlinear:  # TODO: deprecate and/or refactor
    """`order` specifies order of blocks:
        - `-`: weight
        - `N`: normalization
        - `A`: activation
    """
    order: str = '-NA'

    @classmethod
    def new(cls,
            module: type,
            cin: int,
            cout: int = 0,
            *,
            order: str = '',
            **kwargs: object) -> nn.Module:
        order = order or cls.order
        assert {*order} <= {*'AN-'}
        cout = cout or cin

        def _to_layer(char):
            if char == 'N':
                if order.index('N') < order.index('-'):
                    return nn.BatchNorm2d(cin)
                else:
                    return nn.BatchNorm2d(cout)
            elif char == 'A':
                return Activation.new()

            bias = order.index('-') < order.index('N')
            return module(cin, cout, **kwargs, bias=bias)

        if len(order) == 1:
            return _to_layer(order)
        return nn.Sequential(*(_to_layer(o) for o in order))


def linear(cin, cout=0, **kwargs):
    return Nonlinear.new(nn.Linear, cin, cout, **kwargs)


def conv(cin, cout=0, stride=1, padding=1, **kwargs):
    """
    Convolution. Special cases:
        - Channelwise: `padding` = 0, `stride` = 1

    Kernel size equals to `stride + 2 * padding` for integer scaling
    """
    return Nonlinear.new(
        nn.Conv2d,
        cin,
        cout,
        kernel_size=(stride + 2 * padding),
        stride=stride,
        padding=padding,
        **kwargs)


class Cat(nn.Sequential):  # TODO: deprecate and/or refactor
    """
    Helper for U-Net-like modules

    >>> conv = nn.Conv1d(4, 4, 1)
    >>> cat = Cat(conv)
    >>> x = torch.randn(1, 4, 16)
    >>> torch.equal(cat(x), torch.cat([x, conv(x)], dim=1))
    True
    """
    def forward(self, x):
        return torch.cat([x, super().forward(x)], dim=1)


class Sum(nn.Sequential):  # TODO: deprecate and/or refactor
    """Helper for ResNet-like modules"""
    kind = 'resnet'
    expansion: float | None = None
    groups: int | None = None
    blending = False

    def __init__(self,
                 *children: nn.Module,
                 tail: nn.Module = None,
                 ident: nn.Module = None,
                 skip: float = 0.0) -> None:
        super().__init__(*children)
        self.tail = tail
        self.ident = ident
        self.skip = skip

    def forward(self, x):
        y = self.ident(x) if self.ident else x
        if not self.training or self.skip == 0 or self.skip < random.random():
            y = super().forward(x).add_(y)
        return self.tail(y) if self.tail else y

    @classmethod
    def _base_2_way(cls, cin):
        children = [conv(cin), conv(cin, order='-N')]
        return cls(*children, tail=Activation.new(inplace=False))

    @classmethod
    def _base_3_way(cls, cin, expansion, **kwargs):
        mid = int(cin * expansion)
        children = [
            conv(cin, mid, padding=0),
            conv(mid, mid, **kwargs),
            conv(mid, cin, padding=0, order='-N'),
        ]
        if cls.blending:
            children.append(SEBlock.new(cin))
        return cls(*children, tail=Activation.new(inplace=False))

    @classmethod
    def new(cls, cin):
        factory = {
            'resnet': cls._resnet,
            'resnext': cls._resnext,
            'mobile': cls._mobile,
        }.get(cls.kind)
        if factory is None:
            raise ValueError(f'Unsupported {cls.kind}')
        return factory(cin)

    @classmethod
    def _resnet(cls, cin):
        expansion = cls.expansion or (1 / 4)
        return cls._base_3_way(cin, expansion=expansion)

    @classmethod
    def _resnext(cls, cin):
        expansion = cls.expansion or (1 / 2)
        groups = cls.groups or 32
        return cls._base_3_way(cin, expansion=expansion, groups=groups)

    @classmethod
    def _mobile(cls, cin):
        expansion = cls.expansion or 6
        return cls._base_3_way(
            cin, expansion=expansion, groups=cin * expansion)


# -------------------------------- factories --------------------------------


class DenseBlock(nn.Sequential):
    def __init__(self, cin, depth=4, step=16, full=False):
        super().__init__(*(conv(cin + step * i, step, order='NA-')
                           for i in range(depth)))
        self.full = full

    def __repr__(self):
        convs = [
            c for m in self.children()
            for c in m.children() if isinstance(c, nn.modules.conv._ConvNd)
        ]
        cin = convs[0].in_channels
        cout = sum(c.out_channels for c in convs)
        if self.full:
            cout += cin
        return f'{type(self).__name__}({cin}, {cout}, full={self.full})'

    def forward(self, x):
        ys = []
        for module in self.children():
            y = module(x)
            ys.append(y)
            x = torch.cat([x, y], dim=1)
        return x if self.full else torch.cat(ys, dim=1)


class SEBlock(nn.Sequential):
    reduction = 16

    @classmethod
    def new(cls, cin):
        return cls(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, cin // cls.reduction, 1, bias=False),
            Activation.new(),
            nn.Conv2d(cin // cls.reduction, cin, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * super().forward(x)
