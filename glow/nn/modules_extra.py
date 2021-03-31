__all__ = ()

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _CatConv(nn.Sequential):  # a-la DenseNet
    def __init__(self, cin, step, kernel=3):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, step, kernel, padding=padding, bias=False),
        )

    def forward(self, *xs):
        return super().forward(torch.cat(xs, dim=1))


class _Bottleneck(nn.ModuleDict):  # a-la DenseNet-B
    expansion = 4

    def __init__(self, cin, step, kernel=3):
        bnck = step * self.expansion
        padding = (kernel - 1) // 2
        super().__init__({
            'conv1':
                _CatConv(cin, bnck, kernel=1),
            'conv2':
                nn.Sequential(
                    nn.BatchNorm2d(bnck),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(bnck, step, kernel, padding=padding, bias=False),
                ),
        })

    def forward(self, *xs):
        if any(x.requires_grad for x in xs):
            bottleneck = checkpoint(self.conv1, *xs)
        else:
            bottleneck = self.conv1(*xs)
        return self.conv2(bottleneck)


class DenseBlock(nn.Sequential):
    def __init__(self, cin, depth, step, kernel=3, bottleneck=True):
        factory = _Bottleneck if bottleneck else _CatConv
        super().__init__(*(factory(cin + i * step, step, kernel=kernel)
                           for i in range(depth)))

    def forward(self, x):
        xs = [x]
        for layer in self.children():
            xs.append(layer(*xs))
        return torch.cat(xs, dim=1)
