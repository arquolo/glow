__all__ = ('lock_seed', 'param_count')

import random

import numpy as np
import torch
from torch import nn

from ..core import Size

_INT_MAX = 2 ** 32


def param_count(module: nn.Module) -> Size:
    return Size(sum(p.numel() for p in module.parameters()), base=1000)


def lock_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(random.randrange(_INT_MAX))
    torch.manual_seed(random.randrange(_INT_MAX))
