from typing import overload

import torch
from torch import nn, optim


class OptContext:
    def zero_grad(self) -> None:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def step(self) -> None:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        ...

    def __enter__(self) -> 'OptContext':
        ...

    def __exit__(self, type_, *_) -> None:
        ...


@overload
def get_amp_context(net: nn.Module,
                    opt: optim.Optimizer,
                    fp16: bool = ...,
                    retry_on_inf: bool = ...) -> OptContext:
    ...


@overload
def get_amp_context(net: nn.Module,
                    *,
                    fp16: bool = ...,
                    retry_on_inf: bool = ...) -> None:
    ...
