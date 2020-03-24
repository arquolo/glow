__all__ = ('device', 'frozen', 'inference', 'param_count', 'profile')

import functools
from typing import Callable, Iterator, TypeVar, cast
from contextlib import ExitStack, contextmanager

import torch
from torch import nn

from ..core import Size

_F = TypeVar('_F', bound=Callable[..., Iterator])


def device() -> torch.device:
    """Gets current device, including CPU"""
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device('cpu')


@contextmanager
def frozen(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while training.

    Net is switched to eval mode, and its parameters are detached from graph.
    Grads are computed if inputs require grad.
    Works as context manager"""
    with ExitStack() as stack:
        stack.enter_context(torch.onnx.set_training(net, False))
        for p in net.parameters():
            if p.requires_grad:
                stack.callback(p.requires_grad_)
                p.detach_()
        yield


@contextmanager
def inference(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while inference.

    Net is switched to eval mode, and grads' computation is turned off.
    Works as context manager"""
    with torch.onnx.set_training(net, False):
        with torch.no_grad():
            yield


def param_count(net: nn.Module) -> Size:
    """Count of parameters in `net`, both training and not"""
    return Size(sum(p.numel() for p in net.parameters()), base=1000)


def profile(fn: _F) -> _F:
    """Decorator to profile CUDA ops. Use with `nvprof`

    Use in script launched via
    `nvprof --profile-from-start off -o trace.prof -- python main.py`

    >>> @profile
    ... def train_loop():
    ...     for data in loader:
    ...         yield step(data)

    """
    def wrapper(*args, **kwargs):
        results = fn(*args, **kwargs)
        with torch.cuda.profiler.profile():
            yield next(results)
            with torch.autograd.profiler.emit_nvtx():
                yield from results

    return cast(_F, functools.update_wrapper(wrapper, fn))
