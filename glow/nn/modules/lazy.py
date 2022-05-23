__all__ = [
    'LazyConv1d', 'LazyConv2d', 'LazyConv3d', 'LazyConv2dWs',
    'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d',
    'LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyGroupNorm',
    'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d',
    'LazyLayerNorm', 'LazyLinear'
]

import warnings

import torch
from torch import nn
from torch.nn import (
    LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d, LazyConv1d, LazyConv2d,
    LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d,
    LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d, LazyLinear)

from .simple import Conv2dWs

warnings.filterwarnings('ignore', module='torch.nn.modules.lazy')


class _LazyModuleMixin(nn.modules.lazy.LazyModuleMixin):
    def _lazy_load_hook(self: nn.modules.lazy._LazyProtocol, *args, **kwargs):
        super()._lazy_load_hook(*args, **kwargs)  # type: ignore

        # By default, _lazy_load_hook doen't mutate class,
        # if all parameters are initialized from state_dict.
        # Thus, the modules those have all data to instantiate, don't do that.
        # This override fixes that.
        # No more need to call _infer_parameters for fully loaded module.

        # Code from _infer_parameters
        if not self.has_uninitialized_params():  # type: ignore
            self._initialize_hook.remove()
            self._load_hook.remove()
            delattr(self, '_initialize_hook')
            delattr(self, '_load_hook')
            if self.cls_to_become is not None:  # type: ignore
                self.__class__ = self.cls_to_become  # type: ignore


for _tp in [
        LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d, LazyConv1d,
        LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d,
        LazyConvTranspose3d, LazyInstanceNorm1d, LazyInstanceNorm2d,
        LazyInstanceNorm3d, LazyLinear
]:
    globals()[_tp.__name__] = type(_tp.__name__, (_LazyModuleMixin, _tp), {})


class _LazyNormBase(_LazyModuleMixin):
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _set_shape(self, shape: torch.Size) -> None:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        if (not self.has_uninitialized_params()  # type: ignore
                and all(self._get_shape())):
            super().reset_parameters()  # type: ignore

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():  # type: ignore
            self._set_shape(input.shape)
            for p in self.parameters():  # type: ignore
                assert isinstance(p, nn.UninitializedParameter)
                p.materialize(self._get_shape())
            self.reset_parameters()


class LazyLayerNorm(_LazyNormBase, nn.LayerNorm):
    cls_to_become = nn.LayerNorm  # type: ignore[assignment]

    weight: nn.UninitializedParameter  # type: ignore[assignment]
    bias: nn.UninitializedParameter  # type: ignore[assignment]

    def __init__(self,
                 rank: int = 1,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__([0] * rank, eps, False)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.UninitializedParameter()
            self.bias = nn.UninitializedParameter()

    def _get_shape(self) -> tuple[int, ...]:
        return self.normalized_shape

    def _set_shape(self, shape: torch.Size) -> None:
        self.normalized_shape = shape[-len(self.normalized_shape):]


class LazyGroupNorm(_LazyNormBase, nn.GroupNorm):
    cls_to_become = nn.GroupNorm  # type: ignore[assignment]

    weight: nn.UninitializedParameter  # type: ignore[assignment]
    bias: nn.UninitializedParameter  # type: ignore[assignment]

    def __init__(self,
                 num_groups: int,
                 eps: float = 1e-5,
                 affine: bool = True):
        super().__init__(num_groups, 0, eps, False)
        self.affine = affine
        if self.affine:
            self.weight = nn.UninitializedParameter()
            self.bias = nn.UninitializedParameter()

    def _get_shape(self) -> tuple[int, ...]:
        return (self.num_channels, )

    def _set_shape(self, shape: torch.Size) -> None:
        self.num_channels = shape[1]


class LazyConv2dWs(nn.LazyConv2d):
    cls_to_become = Conv2dWs  # type: ignore[assignment]
