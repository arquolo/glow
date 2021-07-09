from __future__ import annotations

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']

import torch
from torch import nn

from .modules import View


def _round8(v, divisor=8):
    """Ensure that number rounded to nearest 8, and error is less than 10%"""
    n = v / divisor
    return int(max(n + 0.5, n * 0.9 + 1)) * divisor


# --------------------------- some wrappers ---------------------------


class Mul(nn.Sequential):
    def forward(self, x):
        return x * super().forward(x)


class Add(nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)


class AsFlat(nn.Sequential):
    """Applies sequence of modules to flattened input"""
    def forward(self, x):
        view = [x.shape[0], -1]
        return super().forward(x.view(view)).view(view + [1] * (x.ndim - 2))


class Conv2dWs(nn.Conv2d):
    """Conv2d block with Weight Standartization

    arXiv preprint arXiv:1903.10520
    """
    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        weight = self.weight
        std, mean = torch.std_mean(weight, (1, 2, 3), keepdim=True)
        weight = (weight - mean) / (std + 1e-5)
        return self._conv_forward(x, weight, self.bias)


# ------------------------------ blocks -------------------------------


def se_block(cin, reduction=4):
    mid = _round8(cin // reduction)
    return Mul(
        nn.AdaptiveAvgPool2d(1),
        AsFlat(
            nn.Linear(cin, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, cin),
        ),
        nn.Hardsigmoid(inplace=True),
    )


def conv_3x3_bn(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),
    )


def conv_1x1_bn(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),
    )


def inv_residual(cin, mid, cout, ksize, stride, se=False, hs=False):
    assert stride in [1, 2]

    activation = (nn.Hardswish if hs else nn.ReLU)(inplace=True)
    block = Add() if stride == 1 and cin == cout else nn.Sequential()

    if cin != mid:
        block.pw = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            activation,
        )

    block.dw = nn.Sequential(
        nn.Conv2d(
            mid, mid, ksize, stride, (ksize - 1) // 2, groups=mid, bias=False),
        nn.BatchNorm2d(mid),
        se_block(mid) if se else nn.Identity(),
        activation,
    )
    block.pw_line = nn.Sequential(
        nn.Conv2d(mid, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
    )
    return block


class MobileNetV3(nn.Sequential):
    """
    Creates a MobileNetV3 Model as defined in:
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen,
    Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan,
    Quoc V. Le, Hartwig Adam. (2019).
    Searching for MobileNetV3
    arXiv preprint arXiv:1905.02244.
    """
    def __init__(self, cfgs: list[list], mode, num_classes=1000, width=1.):
        assert mode in ['large', 'small']
        assert cfgs
        super().__init__()

        cin = _round8(16 * width)
        layers = [conv_3x3_bn(3, cin, 2)]

        mid: int = 0
        for ksize, stride, se, hs, c, t in cfgs:
            cout = _round8(c * width)
            mid = _round8(cin * t)
            layers.append(
                inv_residual(cin, mid, cout, ksize, stride, se=se, hs=hs))
            cin = cout
        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(cin, mid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        cout = {'large': 1280, 'small': 1024}[mode]
        cout = _round8(cout * width) if width > 1.0 else cout
        self.classifier = nn.Sequential(
            View(-1),
            nn.Linear(mid, cout),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(cout, num_classes),
        )

        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.normal_(0, (2. / n) ** 0.5)
                else:
                    m.weight.normal_(0, 0.01)

                if m.bias is not None:
                    m.bias.zero_()


def mobilenetv3_large(**kwargs):
    cfgs = [
        # k, s, SE, HS, c, t
        [3, 1, 0, 0, 16, 1],
        [3, 2, 0, 0, 24, 4],
        [3, 1, 0, 0, 24, 3],
        [5, 2, 1, 0, 40, 3],
        [5, 1, 1, 0, 40, 3],
        [5, 1, 1, 0, 40, 3],
        [3, 2, 0, 1, 80, 6],
        [3, 1, 0, 1, 80, 2.5],
        [3, 1, 0, 1, 80, 2.3],
        [3, 1, 0, 1, 80, 2.3],
        [3, 1, 1, 1, 112, 6],
        [3, 1, 1, 1, 112, 6],
        [5, 2, 1, 1, 160, 6],
        [5, 1, 1, 1, 160, 6],
        [5, 1, 1, 1, 160, 6],
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    cfgs = [
        # k, s, SE, HS, c, t
        [3, 2, 1, 0, 16, 1],
        [3, 2, 0, 0, 24, 4.5],
        [3, 1, 0, 0, 24, 3.67],
        [5, 2, 1, 1, 40, 4],
        [5, 1, 1, 1, 40, 6],
        [5, 1, 1, 1, 40, 6],
        [5, 1, 1, 1, 48, 3],
        [5, 1, 1, 1, 48, 3],
        [5, 2, 1, 1, 96, 6],
        [5, 1, 1, 1, 96, 6],
        [5, 1, 1, 1, 96, 6],
    ]
    return MobileNetV3(cfgs, mode='small', **kwargs)
