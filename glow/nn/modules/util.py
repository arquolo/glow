__all__ = ['NameMixin', 'ActivationFn', 'LazyConvFn', 'LazyNormFn']

from typing import Protocol, TypeVar, Union

from torch import nn

_T = TypeVar('_T')


def pair(t: Union[_T, tuple[_T, ...]]) -> tuple[_T, ...]:
    return t if isinstance(t, tuple) else (t, t)


class NameMixin:
    __constants__ = ['name']
    name: str

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return self.name


class LazyConvFn(Protocol):
    def __call__(self,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True) -> nn.modules.conv._ConvNd:
        ...


class ActivationFn(Protocol):
    def __call__(self, inplace: bool = ...) -> nn.Module:
        ...


class LazyNormFn(Protocol):
    def __call__(self) -> nn.Module:
        ...
