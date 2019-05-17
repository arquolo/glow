import random
from functools import partial

import torch as T
import torch.nn.functional as F
from torch.nn import Module, Sequential

from ..tool import export

# --------------------------------- helpers ---------------------------------


def _deserialize(name):
    fn = getattr(F, name) if isinstance(name, str) else name
    if not callable(fn):
        raise TypeError(f'torch.nn.functional.{name} must be callable')
    return fn


@export
class Lambda(Module):
    def __init__(self, fn, **kwargs):
        super().__init__()
        self.fn = partial(_deserialize(fn), **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.fn(x)


@export
class View(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):  # pylint: disable=arguments-differ
        return x.view(x.shape[0], *self.shape)


@export
class Cat(Sequential):
    """Helper for U-Net-like modules"""
    def forward(self, x):  # pylint: disable=arguments-differ
        return T.cat([module(x) for module in self.children()],
                     dim=1)


@export
class Sum(Sequential):
    """Helper for ResNet-like modules"""

    def __init__(self, *residuals, base=None, skip=0.):
        if base is None:
            base = Sequential()
        super().__init__(base, *residuals)
        self.skip = skip

    def forward(self, x):  # pylint: disable=arguments-differ
        base, *residuals = self.children()
        if self.training and self.skip:
            if self.skip >= random.random():
                return base(x)
        return sum((m(x) for m in residuals),
                   base(x))


@export
class Show(Module):
    """Shows contents of tensors during forward pass"""

    def __init__(self, colored=False):
        super().__init__()
        self.colored = colored

    def forward(self, x):  # pylint: disable=arguments-differ
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

        cv2.imshow(self.__class__.__name__, y)
        cv2.waitKey(16)
        return x

    def __del__(self):
        import cv2
        cv2.destroyWindow(self.__class__.__name__)


@export
class Noise(Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if not self.training:
            return x
        return T.empty_like(x).normal_(std=self.std).add_(x).clamp_(0, 1)
