from __future__ import annotations

__all__ = ['AdamW', 'RAdam', 'Lamb', 'SGDW']

import abc
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import optimizer

# TODO:
# Define Optimizer (as base class) with step(), state_dict(), load_state_dict()
#   targeted to optimization of only one parameter group
#
# Define specific optimizers for each case (SGD, Adam, etc.).
# Override only __init__() and step() methods,
# the rest are `final` in base class
#
# Define Compose as Optimizer to pass options to each sub-optimizer


class _Optimizer(optimizer.Optimizer):
    _step = 0

    def __getstate__(self):
        return {**super().__getstate__(), '_step': self._step}

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = torch.enable_grad()(closure)() if closure is not None else None

        self._step += 1
        for group in self.param_groups:
            kwargs = self._prepare_group(**group)
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Sparse grads are not supported')
                self._do_step(p, self.state[p], **group, **kwargs)

        return loss

    def _prepare_group(self, **group) -> dict[str, Any]:
        return {}

    @abc.abstractmethod
    def _do_step(self, p: torch.Tensor, state: dict[str, torch.Tensor],
                 **group) -> None:
        raise NotImplementedError


def _apply_weight_decay(p: torch.Tensor, lr: float, weight_decay: float,
                        **kwargs):
    if weight_decay:
        p.mul_(1 - lr * weight_decay)


class SGDW(_Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    """
    def __init__(self,
                 params: Iterable[torch.Tensor] | Iterable[dict],
                 lr=0.003,
                 momentum=0.0,
                 dampening=0.0,
                 weight_decay=0.0,
                 nesterov=False):
        for k, v in zip(['lr', 'momentum', 'weight_decay'],
                        [lr, momentum, weight_decay]):
            if v < 0:
                raise ValueError(f'{k} should be greater or equal than zero')
        if nesterov and (momentum == 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _do_step(p: torch.Tensor, state: dict[str, torch.Tensor], **group):
        _apply_weight_decay(p, **group)

        if momentum := group['momentum']:
            if not state:
                avg = state['avg'] = p.grad.clone().detach_()
            else:
                avg = state['avg']
                avg.mul_(momentum).add_(p.grad, alpha=1 - group['dampening'])

            grad = p.grad.add(
                avg, alpha=momentum) if group['nesterov'] else avg
        else:
            grad = p.grad

        p.add_(grad, alpha=-group['lr'])


class AdamW(_Optimizer):
    r"""Implements AdamW algorithm.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params: Iterable[torch.Tensor] | Iterable[dict],
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False) -> None:
        assert 0.0 <= lr
        assert 0.0 <= eps
        for i, beta in enumerate(betas):
            assert 0.0 <= beta < 1.0, f'Invalid beta at index {i}: {betas}'
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        }
        super().__init__(params, defaults)

    def _prepare_group(self, **group) -> dict[str, Any]:
        beta1_t, beta2_t = (beta ** self._step for beta in group['betas'])
        correction1 = 1 - beta1_t
        correction2 = 1 - beta2_t
        return {'step_size': group['lr'] * (correction2 ** 0.5) / correction1}

    @staticmethod
    def _do_step(p: torch.Tensor, state: dict[str, torch.Tensor], **group):
        if p.grad is None:
            return
        _apply_weight_decay(p, **group)

        if not state:
            state['avg'] = torch.zeros_like(p)
            state['avg_sq'] = torch.zeros_like(p)
            if group['amsgrad']:
                state['max_avg_sq'] = torch.zeros_like(p)

        avg, avg_sq = state['avg'], state['avg_sq']
        for avg_, grad, beta in zip([avg, avg_sq], [p.grad, p.grad * p.grad],
                                    group['betas']):
            avg_.lerp_(grad, 1 - beta)

        if group['amsgrad']:
            max_avg_sq = state['max_avg_sq']
            torch.max(max_avg_sq, avg_sq, out=max_avg_sq)
            denom = max_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = avg_sq.sqrt().add_(group['eps'])

        p.addcdiv_(avg, denom, value=-group['step_size'])


class RAdam(_Optimizer):
    def __init__(self,
                 params: Iterable[torch.Tensor] | Iterable[dict],
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0,
                 decay_to_sgd=True) -> None:
        assert 0.0 <= lr
        assert 0.0 <= eps
        defaults = {
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'weight_decay': weight_decay,
            'decay_to_sgd': decay_to_sgd,
        }
        super().__init__(params, defaults)

    def _prepare_group(self, **group) -> dict[str, Any]:
        beta1_t, beta2_t = (beta ** self._step for beta in group['betas'])
        correction1 = 1 - beta1_t
        correction2 = 1 - beta2_t

        _, beta2 = group['betas']
        n_sma_max = 1 / (1 - beta2)
        n_sma = n_sma_max - self._step * beta2_t / correction2

        # More conservative since it's an approximated value
        # Variance is not tractable
        if n_sma < 2.5:
            return {
                'is_tractable': False,
                'step_size': group['lr'] / correction1  # SGD step
            }

        factor = (((n_sma - 2.5) * (n_sma - 1.5) / (n_sma - 0.5)) /
                  ((n_sma_max - 2.5) * (n_sma_max - 1.5) / (n_sma_max - 0.5)))
        return {
            'is_tractable':
                True,
            'step_size':
                group['lr'] * (correction2 * factor) ** 0.5 / correction1,
        }

    @staticmethod
    def _do_step(p: torch.Tensor, state: dict[str, torch.Tensor], **group):
        _apply_weight_decay(p, **group)

        if not state:
            state['avg'] = torch.zeros_like(p)
            state['avg_sq'] = torch.zeros_like(p)

        avg, avg_sq = state['avg'], state['avg_sq']
        for avg_, grad, beta in zip([avg, avg_sq], [p.grad, p.grad * p.grad],
                                    group['betas']):
            avg_.lerp_(grad, 1 - beta)

        if group['is_tractable']:
            denom = avg_sq.sqrt().add_(group['eps'])
            p.addcdiv_(avg, denom, value=-group['step_size'])

        elif group['decay_to_sgd']:
            p.add_(avg, alpha=-group['step_size'])


class Lamb(_Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)

    __ https://arxiv.org/abs/1904.00962
    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """
    def __init__(self,
                 params: Iterable[torch.Tensor] | Iterable[dict],
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 clamp_value=10,
                 adam=False,
                 debias=False):
        assert lr >= 0
        assert eps >= 0
        assert clamp_value >= 0
        assert weight_decay >= 0
        for i, beta in enumerate(betas):
            assert 0.0 <= beta < 1.0, f'Invalid beta at index {i}: {betas}'

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'debias': debias,
            'clamp_value': clamp_value,
            'adam': adam,
        }
        super().__init__(params, defaults)

    def _prepare_group(self, **group) -> dict[str, Any]:
        if group['debias']:
            return {'step_size': group['lr']}

        beta1_t, beta2_t = (beta ** self._step for beta in group['betas'])
        correction1 = 1 - beta1_t
        correction2 = 1 - beta2_t
        return {'step_size': group['lr'] * (correction2 ** 0.5) / correction1}

    @staticmethod
    def _do_step(p: torch.Tensor, state: dict[str, torch.Tensor], **group):
        if not state:
            state['avg'] = torch.zeros_like(p)
            state['avg_sq'] = torch.zeros_like(p)

        avg, avg_sq = state['avg'], state['avg_sq']
        for avg_, grad, beta in zip([avg, avg_sq], [p.grad, p.grad * p.grad],
                                    group['betas']):
            avg_.lerp_(grad, 1 - beta)

        weight_norm = torch.norm(p).clamp_(0, group['clamp_value'])

        step = avg / avg_sq.sqrt().add_(group['eps'])
        if weight_decay := group['weight_decay']:
            step.add_(p, alpha=weight_decay)

        norm = torch.norm(step)
        ratio = 1. if group['adam'] else torch.where(
            (weight_norm * norm).bool(),
            weight_norm / norm,
            torch.as_tensor(1).to(p.device),
        )
        p.add_(step, alpha=-group['step_size'] * ratio)


"""
optimizer -> group:dataclass -> __init__() -> update(p)
"""
from dataclasses import InitVar, dataclass, field


@dataclass
class _IOptimizer:
    params: InitVar[Iterable[torch.Tensor] | Iterable[dict]]

    _groups: list[Optimizer] = field(init=False, default_factory=list)
    _params: list[torch.Tensor] = field(init=False, default_factory=list)
    _state: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self, params):
        *params, = params
        defaults = {
            k: v for k, v in vars(self).items()
            if k not in ('_groups', '_params', '_state')
        }
        if params and isinstance(params[0], dict):
            self._groups.extend(
                type(self)(group['params'], **{
                    **defaults,
                    **{k: v for k, v in group.items() if k != 'params'}
                }) for group in params)
        else:
            self._params = params


@dataclass
class Adam(_IOptimizer):
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults

        if isinstance(params, torch.Tensor):
            raise TypeError('params argument given to the optimizer should be '
                            'an iterable of Tensors or dicts, but got ' +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if not param_groups:
            raise ValueError('optimizer got an empty parameter list')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        lines = [f'{type(self).__name__} (']
        for i, group in enumerate(self.param_groups):
            lines += [f'Parameter Group {i}']
            lines += [
                f'    {key}: {value}'
                for key, value in sorted(group.items()) if key != 'params'
            ]
        lines += [')']
        return '\n'.join(lines)

    def state_dict(self):
        mappings = {
            id(p): i for group in self.param_groups
            for i, p in enumerate(group['params'])
        }
        groups = [{
            **{k: v for k, v in group.items() if k != 'params'},
            'params': [mappings[id(p)] for p in group['params']]
        } for group in self.param_groups]
        state = {(mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                 for k, v in self.state.items()}
        return {'state': state, 'param_groups': groups}

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        assert [len(g['params']) for g in groups] \
            == [len(g['params']) for g in saved_groups]

        id_map = {
            old_id: p for old_id, p in zip(
                chain.from_iterable((g['params'] for g in saved_groups)),
                chain.from_iterable((g['params'] for g in groups)))
        }

        def cast(param, value):
            if isinstance(value, torch.Tensor):
                if param.is_floating_point():
                    value = value.to(param.dtype)
                return value.to(param.device)
            if isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            if isinstance(value, Iterable):
                return type(value)(cast(param, v) for v in value)
            return value

        state = dict([((param, cast(param, v)) if
                       (param := id_map.get(k)) is not None else (k, v))
                      for k, v in state_dict['state'].items()])

        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)
        ]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or set_to_none:
                    p.grad = None
                    continue
                if p.grad.grad_fn is not None:
                    p.grad.detach_().zero_()
                else:
                    p.grad.requires_grad_(False).zero_()

    def step(self, closure):
        raise NotImplementedError

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict)

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        else:
            assert not isinstance(params, set)
            param_group['params'] = list(params)

        for param in param_group['params']:
            assert isinstance(param, torch.Tensor)
            assert param.is_leaf

        for name, default in self.defaults.items():
            assert default is not required or name in param_group
            param_group.setdefault(name, default)

        params = param_group['params']
        assert len(params) == len(set(params))

        param_set = {p for group in self.param_groups for p in group['params']}
        assert not param_set & {*param_group['params']}

        self.param_groups.append(param_group)
