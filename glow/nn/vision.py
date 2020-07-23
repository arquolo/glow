__all__ = ['Show']

import weakref

import cv2
import numpy as np
import torch
from torch import nn


# TODO: rewrite like `def traced(nn.Module) -> nn.Module`
class Show(nn.Module):
    """Shows contents of tensors during forward pass"""
    sigmas = 2

    def __init__(self, colored=False):
        super().__init__()
        self.name = f'{type(self).__name__}_0x{id(self):x}'
        self.colored = colored
        self.close = weakref.finalize(self, cv2.destroyWindow, self.name)

    def forward(self, inputs: torch.Tensor):
        bs, ch, h, w = inputs.shape

        ten = torch.tensor(inputs)

        bias = ten.mean([-2, -1], keepdim=True)
        scale = ten.std([-2, -1], keepdim=True)
        scale = scale.mul_(self.sigmas / 128).clamp_(1e-5).inverse()

        ten = ten.sub_(bias).mul_(scale).add_(128).clamp_(0, 255)
        image: np.ndarray = ten.byte().cpu().numpy()

        # TODO: test numpy.reshape vs torch.permute
        if self.colored:
            groups = ch // 3
            image = image[:, :groups * 3, :, :].reshape(bs, -1, 3, h, w)
            image = image.transpose(0, 3, 1, 4, 2).reshape(bs * h, -1, 3)
        else:
            image = image.transpose(0, 2, 1, 3).reshape(bs * h, -1)

        cv2.imshow(self.name, image)
        cv2.waitKey(1)

        return inputs
