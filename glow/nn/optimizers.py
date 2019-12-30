__all__ = ('RAdam', )

import torch
from torch.optim import optimizer


class RAdam(optimizer.Optimizer):
    _bufsize = 10

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
        self._step = 0
        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def __getstate__(self):
        return {**super().__getstate__(), 'step': self._step}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._step += 1
        for group in self.param_groups:
            n_sma, step_size = self._update_group(group)
            for p in group['params']:
                self._do_step(p, group, n_sma=n_sma, step_size=step_size)

        return loss  # noqa: R504

    def _update_group(self, group):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1**self._step
        bias_correction2 = 1 - beta2**self._step

        beta2_t = beta2**self._step
        n_sma_max = 2 / (1 - beta2) - 1
        n_sma = n_sma_max - 2 * self._step * beta2_t / bias_correction2

        # more conservative since it's an approximated value
        step_size = 1
        if n_sma >= 5:
            k = (n_sma - 4) * (n_sma - 2) / n_sma
            k_max = (n_sma_max - 4) * (n_sma_max - 2) / n_sma_max
            step_size = ((1 - beta2_t) * k / k_max)**.5

        return n_sma, (step_size / bias_correction1)

    def _do_step(self, p, group, n_sma, step_size):
        if p.grad is None:
            return
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('RAdam does not support sparse gradients')

        state: dict = self.state[p]
        if not state:
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        if group['weight_decay'] != 0:
            p.data.add_(-group['weight_decay'] * group['lr'], p.data)

        # more conservative since it's an approximated value
        if n_sma >= 5:
            denom = exp_avg_sq.sqrt().add_(1e-8)
            p.data.addcdiv_(-step_size * group['lr'], exp_avg, denom)
        else:
            p.data.add_(-step_size * group['lr'], exp_avg)