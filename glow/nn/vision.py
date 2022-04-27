__all__ = ['Show']

import weakref

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# TODO: rewrite like `def traced(nn.Module) -> nn.Module`
# TODO: use pyqt/matplotlib to create window


class Show(nn.Module):
    """Shows contents of tensors during forward pass"""
    nsigmas = 2
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, colored: bool = False):
        super().__init__()
        self.name = f'{type(self).__name__}_0x{id(self):x}'
        self.colored = colored
        self.register_buffer(
            'weight', torch.tensor(128 / self.nsigmas), persistent=False)
        self.register_buffer('bias', torch.tensor(128.), persistent=False)

        self.close = weakref.finalize(self, cv2.destroyWindow, self.name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, ch, _, _ = inputs.shape

        ten = torch.tensor(inputs)

        weight = self.weight.expand(ch)
        bias = self.bias.expand(ch)
        ten = F.instance_norm(ten, weight=weight, bias=bias)
        image: np.ndarray = ten.clamp_(0, 255).byte().cpu().numpy()

        if self.colored:
            groups = ch // 3
            image = image[:, :groups * 3, :, :]
            image = rearrange(image, 'b (g c) h w -> (b h) (g w) c', c=3)
        else:
            image = rearrange(image, 'b c h w -> (b h) (c w)')

        cv2.imshow(self.name, image)
        cv2.waitKey(1)

        return inputs
