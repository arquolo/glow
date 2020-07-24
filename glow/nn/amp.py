"""Utilities to train and eval nn.Module in half-precision mode (float16)."""

__all__ = ['autocast', 'get_amp_context']

import contextlib
import warnings
from collections import abc
from functools import partial
from typing import Dict, Iterable, Tuple, overload

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim

_MIN_SCALE = 2.0 ** -16
_MAX_SCALE = 2.0 ** +16
_PATIENCE = 2000
# _TORCH_1_7 = False
_TORCH_1_7 = torch.__version__ >= '1.7'


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


def deep_to(xs, device: torch.device = None, dtype: torch.dtype = None):
    def fun(x):
        if x.is_floating_point():
            return x.to(device, dtype, non_blocking=True)
        return x.to(device, non_blocking=True)

    return _apply(xs, fun)


class OptContext:
    def __init__(self, optim: torch.optim.Optimizer):  # type: ignore
        self.optim = optim
        self.optim.load_state_dict(self.optim.state_dict())

    def zero_grad(self) -> None:
        self.optim.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        self.optim.step()

    def state_dict(self) -> dict:
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

    def __enter__(self) -> 'OptContext':
        self.zero_grad()
        return self

    def __exit__(self, type_, *_) -> None:
        if type_ is None:
            self.step()


class _AmpContext(OptContext):
    _pending_steps: int = 0

    def __init__(self,
                 optim: torch.optim.Optimizer,
                 master_params: Iterable[nn.Parameter],
                 retries: bool = True):
        self._master_params = master_params
        self._retries = retries

        self._step = torch.zeros(1).int()
        self._scale = torch.empty(1).fill_(_MAX_SCALE)

        self._devices = {
            p.device for param_group in optim.param_groups
            for p in param_group['params'] if p.requires_grad
        }
        super().__init__(optim)

    def zero_grad(self) -> None:
        self._pending_steps = 0

    def _replicate_scale(self) -> Tuple[dict, dict]:
        inv_scale = self._scale.double().reciprocal().float()
        return (
            {dev: inv_scale.to(dev) for dev in self._devices},
            {dev: torch.zeros(1, device=dev) for dev in self._devices},
        )

    def _update_scale_failed(self, infs: dict):
        found_inf = torch.cat(
            [inf.to(self._scale.device) for inf in infs.values()]).sum()
        self._scale = torch._amp_update_scale(  # type: ignore
            self._step, self._scale, found_inf, 2.0, 0.5, _PATIENCE)
        self._scale.clamp_(_MIN_SCALE, _MAX_SCALE)
        return found_inf.item()

    def _do_grad_transfer_and_check(self) -> bool:
        raise NotImplementedError

    def backward(self, loss: torch.Tensor) -> None:
        self._step = self._step.to(loss.device, non_blocking=True)
        self._scale = self._scale.to(loss.device, non_blocking=True)

        while self._scale.item() > _MIN_SCALE:
            for p in self._master_params:  # zero local grads
                if p.grad is not None:
                    p.grad.detach_().zero_()

            # collect grads
            (loss * self._scale).backward(retain_graph=self._retries)

            with torch.no_grad():  # transfer grads and check for inf/nan
                if self._do_grad_transfer_and_check():
                    self._pending_steps += 1
                    return
                if not self._retries:
                    return

        raise OverflowError(f'Cannot decrease scale below {_MIN_SCALE}')


class _AmpContext15(_AmpContext):
    def __init__(
            self,
            optim: torch.optim.Optimizer,  # type: ignore
            retries: bool = True):
        self._mix_to_full: Dict[nn.Parameter, nn.Parameter] = {}

        for param_group in optim.param_groups:
            for i, p in enumerate(param_group['params']):
                if p.requires_grad:
                    p32 = p.to(torch.float32, copy=True)
                    p32.detach_().requires_grad_()
                    self._mix_to_full[p] = param_group['params'][i] = p32
                    if p in optim.state:
                        optim.state[p32] = optim.state.pop(p)

        super().__init__(optim, [*self._mix_to_full], retries=retries)

    def _do_grad_transfer_and_check(self) -> bool:
        inv_scales, infs = self._replicate_scale()

        grads: Dict[nn.Parameter, torch.Tensor] = {}
        for p, p32 in self._mix_to_full.items():  # unscale grads
            if p.grad is None:
                continue
            if p32.grad is None:
                p32.grad = torch.empty_like(p.grad, dtype=torch.float32)
            grads[p32] = p.grad.float()
            torch._amp_non_finite_check_and_unscale_(
                grads[p32], infs[p.device], inv_scales[p.device])

        if self._update_scale_failed(infs):
            return False

        for p32, grad in grads.items():
            if p32.grad is not None:
                if not self._pending_steps:
                    p32.grad.set_(grad)
                else:
                    p32.grad.add_(grad)
        return True

    def step(self) -> None:
        if not self._pending_steps:
            return
        super().step()
        with torch.no_grad():
            for p, p32 in self._mix_to_full.items():  # update fp16 parameters
                p.copy_(p32)


class _AmpContext17(_AmpContext):
    def __init__(
            self,
            optim: torch.optim.Optimizer,  # type: ignore
            retries: bool = True) -> None:
        self._grads = {
            p: torch.empty_like(p) for param_group in optim.param_groups
            for p in param_group['params'] if p.requires_grad
        }
        super().__init__(optim, [*self._grads], retries)

    def _do_grad_transfer_and_check(self) -> bool:
        inv_scales, infs = self._replicate_scale()

        for p in self._grads:  # unscale grads
            if p.grad is None:
                continue
            torch._amp_non_finite_check_and_unscale_(  # type: ignore
                p.grad, infs[p.device], inv_scales[p.device])

        if self._update_scale_failed(infs):
            return False

        for p, grad in self._grads.items():  # update buffer grads
            if p.grad is not None:
                if not self._pending_steps:
                    grad.copy_(p.grad)
                else:
                    grad.add_(p.grad)
        return True

    def step(self) -> None:
        if not self._pending_steps:
            return
        with torch.no_grad():
            for p, grad in self._grads.items():  # update master grads
                if p.grad is not None:
                    p.grad.copy_(grad)
        super().step()


def _deep_to_hook(_, xs, device=None, dtype=None):
    return deep_to(xs, device, dtype)


def autocast(enabled: bool = True):
    return (torch.cuda.amp.autocast(enabled=enabled)
            if _TORCH_1_7 else contextlib.nullcontext())


@overload
def get_amp_context(net: nn.Module,
                    opt: torch.optim.Optimizer,
                    *,
                    fp16: bool = False,
                    retries: bool = True) -> OptContext:
    ...


@overload
def get_amp_context(net: nn.Module,
                    *,
                    fp16: bool = False,
                    retries: bool = True) -> None:
    ...


def get_amp_context(net, opt=None, *, fp16=False, retries=True):
    """Switch model and optimizer to mixed precision mode

    Parameters:
      - fp16 - enables fp16 mode.
      - retries - if set, probe another scale for the same loss
        when `backward()` fills grads with NaN/Inf. Only for fp16 mode.

    Example:
    ```
        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        # Before training loop
        ctx = gnn.get_amp_context(model, optimizer, fp16=True)

        for input, target in data:
            with gnn.autocast():
                output = model(input)
            with ctx:  # Enter resets grads, exit updates parameters
                loss = loss_fn(output, target)
                ctx.backward(loss)
    ```

    """
    dtype = torch.float32
    if fp16:
        if all(
                torch.cuda.get_device_capability(dev) < (7, 0)
                for dev in range(torch.cuda.device_count())):
            warnings.warn('Neither of devices support hardware FP16, '
                          "that's why don't expect speed improvement")
        dtype = torch.float16

    # patch inputs
    net.register_forward_pre_hook(partial(_deep_to_hook, dtype=dtype))
    if not _TORCH_1_7:
        net.to(dtype, non_blocking=True)

    # patch normalization
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._NormBase):  # type: ignore
            if m.affine:
                m.float()

    if opt is None:
        return None
    if not fp16:
        return OptContext(opt)
    return (_AmpContext17 if _TORCH_1_7 else _AmpContext15)(opt, retries)


def _warning_on_one_line(message,
                         category,
                         filename,
                         lineno,
                         file=None,
                         line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'


warnings.formatwarning = _warning_on_one_line  # type: ignore
