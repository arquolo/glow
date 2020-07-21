"""
Utilities to train and eval nn.Module in half-precision mode (float16).

Example::

    device = torch.device('cuda')
    net = nn.Linear(16, 4)
    opt = torch.optim.SGD(net.parameters())
    loss_fn = nn.MSELoss()

    # before training loop
    ctx = amp_init_opt(net, opt, device=device)

    # get data
    x = torch.randn(2, 16)  # type/device cast is not needed for inputs
    y = torch.randn(2, 4)

    with ctx:
        # enter resets grads
        out = net(x)
        loss = loss_fn(out, y.to(device))

        ctx.backward(loss)  # do backward pass
        # exit updates weights

"""

__all__ = ('amp_init', 'amp_init_opt')

import functools
import warnings
from collections import abc
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

_MIN_SCALE = 2.0 ** -16
_MAX_SCALE = 2.0 ** +16
_PATIENCE = 2000


def _apply(xs, fn):
    if isinstance(xs, torch.Tensor):
        return fn(xs)
    if isinstance(xs, (str, bytes, np.ndarray)):
        return xs
    if isinstance(xs, abc.Mapping):
        return dict(_apply(kv, fn) for kv in xs.items())
    if isinstance(xs, abc.Iterable):
        return type(xs)(_apply(x, fn) for x in xs)
    return xs


class OptimizerWrapper:
    def __init__(self, optim: torch.optim.Optimizer):  # type: ignore
        self.optim = optim
        self.optim.load_state_dict(self.optim.state_dict())

    def zero_grad(self) -> None:
        self.optim.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        self.optim.step()

    def state_dict(self):
        return {
            k: (v if k != 'optim' else v.state_dict())
            for k, v in self.__dict__.items()
        }

    def load_state_dict(self, state_dict):
        state_dict = {**state_dict}
        self.optim.load_state_dict(state_dict.pop('optim'))

        dst = []
        _apply({k: self.__dict__[k] for k in state_dict}, dst.append)

        with torch.no_grad():
            self.__dict__.update(
                _apply(state_dict, lambda src: dst.pop(0).copy_(src)))

    def __enter__(self) -> 'OptimizerWrapper':
        self.zero_grad()
        return self

    def __exit__(self, type_, *_) -> None:
        if type_ is None:
            self.step()


class _MixedOptimizer(OptimizerWrapper):
    _pending_steps: int = 0

    def __init__(
            self,
            optim: torch.optim.Optimizer,  # type: ignore
            allow_skip: bool = False) -> None:
        self._step = 0
        self._scale = _MAX_SCALE
        self._allow_skip = allow_skip
        self._mix_to_full: Dict[nn.Parameter, nn.Parameter] = {}

        for param_group in optim.param_groups:
            for i, p in enumerate(param_group['params']):
                if p.requires_grad:
                    p32 = p.to(torch.float32, copy=True)
                    p32.detach_().requires_grad_()
                    self._mix_to_full[p] = param_group['params'][i] = p32
                    if p in optim.state:
                        optim.state[p32] = optim.state.pop(p)

        super().__init__(optim)

    def zero_grad(self) -> None:
        self._pending_steps = 0

    def _do_grad_transfer(self) -> bool:
        checks: List[torch.Tensor] = []
        grads: Dict[nn.Parameter, torch.Tensor] = {}

        for p, p32 in self._mix_to_full.items():
            if p.grad is None:
                continue

            # delay check
            if p.dtype == torch.half:
                checks.append(torch.isfinite(p.grad).all())
            if p32.grad is None:
                p32.grad = torch.empty_like(p.grad, dtype=torch.float32)

            # transfer grads from fp16/fp32 to fp32 with scaling
            if not self._pending_steps:
                torch.mul(p.grad, 1 / self._scale, out=p32.grad)
            else:
                grads[p32] = p32.grad.add(1 / self._scale, p.grad)

        # now check
        if not torch.stack(checks).all():
            # nan/inf catched in fp16 grads, reduce scale
            self._step = 0
            self._scale /= 2
            return False

        for p32, grad in grads.items():
            if p32.grad is not None:
                p32.grad.set_(grad)

        self._step += 1
        self._pending_steps += 1

        if self._step >= _PATIENCE and self._scale < _MAX_SCALE:
            self._step = 0
            self._scale *= 2
        return True

    def backward(self, loss: torch.Tensor) -> None:
        retain_graph = not self._allow_skip

        while self._scale > _MIN_SCALE:
            # zero local grads
            for p in self._mix_to_full:
                if p.grad is not None:
                    p.grad.detach_().zero_()

            # collect grads
            (loss * self._scale).backward(retain_graph=retain_graph)

            # transfer and check for correctness
            with torch.no_grad():
                if self._do_grad_transfer() or self._allow_skip:
                    return

        raise OverflowError(f'Cannot decrease scale below {_MIN_SCALE}')

    def step(self) -> None:
        if not self._pending_steps:
            return
        super().step()
        with torch.no_grad():
            for p, p32 in self._mix_to_full.items():  # update fp16 parameters
                p.copy_(p32)


def _to(_, xs, device, dtype):
    def fun(x):
        if x.is_floating_point():
            return x.to(device, dtype, non_blocking=True)
        return x.to(device, non_blocking=True)

    return _apply(xs, fun)


def amp_init(net: nn.Module,
             device: Union[int, torch.device] = 0,
             fp16: bool = False) -> None:
    """Switch model to mixed precision mode if possible"""
    if isinstance(device, int):
        device = torch.device(device)

    dtype = torch.float32
    if fp16:
        if torch.cuda.get_device_capability(device) < (7, ):
            warnings.warn(f'FP16 on {device} is emulated, so no speed benefit')
        dtype = torch.float16

    # patch inputs
    input_caster = functools.partial(_to, device=device, dtype=dtype)
    net.register_forward_pre_hook(input_caster)
    net.to(device, dtype, non_blocking=True)

    # patch normalization
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):  # type: ignore
            if m.affine:
                m.float()


def amp_init_opt(
        net: nn.Module,
        opt: torch.optim.Optimizer,  # type: ignore
        device: Union[int, torch.device] = 0,
        fp16: bool = False,
        allow_skip: bool = False) -> OptimizerWrapper:
    """Switch model and optimizer to mixed precision mode

    Parameters:
      - fp16 - enables fp16 mode
      - allow_skip - skip updates when `backward()` fills grads with NaN/Inf.
        Only for fp16 mode
    """
    amp_init(net, device=device, fp16=fp16)
    if not fp16:
        return OptimizerWrapper(opt)
    return _MixedOptimizer(opt, allow_skip)


def _warning_on_one_line(message,
                         category,
                         filename,
                         lineno,
                         file=None,
                         line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'


warnings.formatwarning = _warning_on_one_line  # type: ignore
