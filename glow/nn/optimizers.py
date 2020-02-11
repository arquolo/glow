__all__ = ('RAdam', )

import torch
from torch.optim import optimizer


class RAdam(optimizer.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0,
                 decay_to_sgd=True) -> None:
        self._step = 0
        self._decay_to_sgd = decay_to_sgd
        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def __getstate__(self):
        return {**super().__getstate__(), '_step': self._step}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._step += 1
        for group in self.param_groups:
            is_tractable, step_size = self._update_group(group)
            for p in group['params']:
                self._do_step(p, group, is_tractable, step_size=step_size)

        return loss  # noqa: R504

    def _update_group(self, group):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** self._step
        bias_correction2 = 1 - beta2 ** self._step

        beta2_t = beta2 ** self._step
        n_sma_max = 2 / (1 - beta2) - 1
        n_sma = n_sma_max - 2 * self._step * beta2_t / bias_correction2

        # more conservative since it's an approximated value
        # variance is not tractable
        if n_sma < 5:
            return False, (1 / bias_correction1)

        k = (n_sma - 4) * (n_sma - 2) / n_sma
        k_max = (n_sma_max - 4) * (n_sma_max - 2) / n_sma_max
        step_size = ((1 - beta2_t) * k / k_max) ** 0.5 / bias_correction1
        return True, step_size

    @torch.no_grad()
    def _do_step(self, p, group, is_tractable, step_size):
        if p.grad is None:
            return
        if p.grad.is_sparse:
            raise RuntimeError('RAdam does not support sparse gradients')

        state: dict = self.state[p]
        if not state:
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg.mul_(beta1).add_(1 - beta1, p.grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, p.grad, p.grad)

        if group['weight_decay'] != 0:
            p.data.mul_(1 - group['weight_decay'] * group['lr'])

        if is_tractable:
            denom = exp_avg_sq.sqrt().add_(1e-8)
            p.addcdiv_(-step_size * group['lr'], exp_avg, denom)
        elif self._decay_to_sgd:
            p.add_(-step_size * group['lr'], exp_avg)
