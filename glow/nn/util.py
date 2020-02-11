__all__ = ('param_count', )

from torch import nn

from ..core import Size


def param_count(module: nn.Module) -> Size:
    return Size(sum(p.numel() for p in module.parameters()), base=1000)
