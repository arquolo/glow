__all__ = ('device', 'frozen', 'param_count', 'profile')

import contextlib
import functools
from typing import Callable, Iterator, TypeVar, cast

import torch
from torch import nn

from ..core import Size

_F = TypeVar('_F', bound=Callable[..., Iterator])


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
