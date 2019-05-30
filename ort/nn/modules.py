import functools
import random

import torch as T
import torch.nn.functional as F
from torch.nn import Module, Sequential

from ..tool import export, pretty_dict
from ..config import capture

# ----------------------------- pure functions -----------------------------


class Wrapper(Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped


@export
@capture(prefix='nn.activation')
class Activation(Wrapper):
    def __init__(self, fn=F.relu):
        super().__init__(functools.partial(fn, inplace=True))

    def __repr__(self):
        fn = self.wrapped.func
        return (f'{type(self).__name__}'
                f'({fn.__module__}.{fn.__qualname__},'
                f' {pretty_dict(self.wrapped.keywords)})')

    def forward(self, x):
        return self.wrapped(x)


@export
class View(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


@export
class Show(Module):
    """Shows contents of tensors during forward pass"""

    def __init__(self, colored=False):
        super().__init__()
        self.colored = colored
        self.name = f'{type(self).__name__}_{id(self):x}'

    def forward(self, x):
        import cv2
        bs, ch, h, w = x.shape

        y = x.clone().requires_grad_().detach()
        y -= y.min(3, keepdim=True).values.min(2, keepdim=True).values
        y /= y.max(3, keepdim=True).values.max(2, keepdim=True).values
        y = y.mul_(255).byte().cpu().numpy()
        if self.colored:
            ch = (ch // 3) * 3
            y = y[:, :ch, :, :].reshape(bs, -1, 3, h, w)
            y = y.transpose(0, 3, 1, 4, 2).reshape(bs * h, ch * w // 3, 3)
        else:
            y = y.transpose(0, 2, 1, 3).reshape(bs * h, ch * w)

        cv2.imshow(self.name, y)
        cv2.waitKey(16)
        return x

    def __del__(self):
        import cv2
        cv2.destroyWindow(self.name)


@export
class Noise(Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if not self.training:
            return x
        return T.empty_like(x).normal_(std=self.std).add_(x).clamp_(0, 1)

# ---------------------------- stateful modules -----------------------------


@capture(prefix='nn.conv')
def norm_activated(module: Module,
                   cin: int,
                   cout: int = 0, *,
                   config='-NA', **kwargs):
    """Config is a sequence of chars, each means:
        - `-`: weight
        - `N`: normalization
        - `A`: activation
    """
    assert set(config) <= set('AN-')
    cout = cout or cin

    def _to_layer(char):
        if char == 'N':
            ch = cin if config.index('N') < config.index('-') else cout
            return T.nn.BatchNorm2d(ch)
        elif char == 'A':
            return Activation()
        else:
            bias = config.index('-') < config.index('N')
            return module(cin, cout, **kwargs, bias=bias)

    return (Sequential(*map(_to_layer, config)) if len(config) > 1
            else _to_layer(config))


@export
def linear(cin, cout=0, **kwargs):
    return norm_activated(T.nn.Linear, cin, cout, **kwargs)


@export
def conv(cin, cout=0, groups=1, stride=1, padding=1, **kwargs):
    """
    Convolution. Special cases:
        - Depthwise: `groups` = 0
        - Channelwise: `padding` = 0, `stride` = 1

    Kernel size equals to `stride + 2 * padding` for integer scaling
    """
    groups = groups or cout or cin
    return norm_activated(
        T.nn.Conv2d, cin, cout,
        kernel_size=(stride + 2 * padding),
        stride=stride, padding=padding, groups=groups, **kwargs)


@export
class Cat(Sequential):
    """Helper for U-Net-like modules"""

    def forward(self, x):
        return T.cat([x, super().forward(x)],
                     dim=1)


@export
class Sum(Wrapper):
    """Helper for ResNet-like modules"""

    def __init__(self, residual, base=None, tail=None, skip=0.):
        super().__init__(residual)
        self.base = base
        self.tail = tail
        self.skip = skip

    def forward(self, x):
        y = self.base(x) if self.base else x
        if not self.training or not self.skip or self.skip < random.random():
            y += self.wrapped(x)
        return self.tail(y) if self.tail else y

    @classmethod
    def resnet_v1(cls, cin):
        return cls(Sequential(conv(cin),
                              conv(cin, config='-N')),
                   tail=Activation())

    @classmethod
    def _resbase(cls, cin, alpha=1, **kwargs):
        mid = int(cin * alpha)
        return cls(Sequential(conv(cin, mid, padding=0),
                              conv(mid, mid, **kwargs),
                              conv(mid, cin, padding=0, config='-N')),
                   tail=Activation())

    @classmethod
    def resnet_v2(cls, cin, compression=4):
        return cls._resbase(cin, alpha=1 / compression)

    @classmethod
    def mobilenet_v2(cls, cin, expansion=6):
        return cls._resbase(cin, alpha=expansion, groups=0)

# -------------------------------- factories --------------------------------


@export
class DenseBlock(Sequential):
    def __init__(self, cin, depth=4, step=16, full=False):
        super().__init__(*(conv(cin + step * i, step, config='NA-')
                         for i in range(depth)))
        self.full = full

    def __repr__(self):
        convs = [c for m in self.children() for c in m.children()
                 if isinstance(c, T.nn.modules.conv._ConvNd)]
        cin = convs[0].in_channels
        cout = sum(c.out_channels for c in convs)
        if self.full:
            cout += cin
        return f'{type(self).__name__}({cin}, {cout}, full={self.full})'

    def forward(self, x):
        xs = []
        for m in self.children():
            out = m(x)
            xs.append(out)
            x = T.cat([x, out], dim=1)
        return x if self.full else T.cat(xs, dim=1)
