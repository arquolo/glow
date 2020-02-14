__all__ = ('device', 'frozen', 'param_count')

import contextlib
from typing import Iterator

import torch
from torch import nn

from ..core import Size


def device() -> torch.device:
    """Gets current device, including CPU"""
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device('cpu')


@contextlib.contextmanager
def frozen(net: nn.Module) -> Iterator[None]:
    """Context manager which frozes `net` turning it to eval mode"""
    with contextlib.ExitStack() as stack:
        stack.callback(net.train, net.training)
        stack.enter_context(torch.no_grad())
        net.eval()
        yield


def param_count(net: nn.Module) -> Size:
    """Count of parameters in `net`, both training and not"""
    return Size(sum(p.numel() for p in net.parameters()), base=1000)
