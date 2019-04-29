import random
from functools import partial

import torch as T
import torch.nn.functional as F
from torch.nn import Module, Sequential

# --------------------------------- helpers ---------------------------------


def _deserialize(name):
    fn = getattr(F, name) if isinstance(name, str) else name
    if not callable(fn):
        raise TypeError(f'torch.nn.functional.{name} must be callable')
    return fn


class Lambda(Module):
    def __init__(self, fn, **kwargs):
        super().__init__()
        self.fn = partial(_deserialize(fn), **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.fn(x)


class View(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):  # pylint: disable=arguments-differ
        return x.view(x.shape[0], *self.shape)


class Cat(Sequential):
    """Helper for U-Net-like modules"""
    def forward(self, x):  # pylint: disable=arguments-differ
        return T.cat([module(x) for module in self.children()],
                     dim=1)


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


class Show(Module):
    """Shows contents of tensors during forward pass"""

    def forward(self, x):  # pylint: disable=arguments-differ
        import cv2
        bs, ch, h, w = x.shape

        arr = x.detach()
        arr -= arr.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
        arr /= arr.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
        arr = arr.mul_(255).byte().cpu()
        arr = arr.numpy().transpose(0, 2, 1, 3).reshape(bs * h, ch * w)

        cv2.imshow('out', arr)
        cv2.waitKey(16)
        return x

    def __del__(self):
        import cv2
        cv2.destroyAllWindows()
