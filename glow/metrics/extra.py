__all__ = ('accuracy_', 'dice')

import torch

from .base import to_index


def accuracy_(pred, true) -> torch.Tensor:
    _, pred, true = to_index(pred, true)
    return (true == pred).double().mean()


def dice(pred, true, macro=False) -> torch.Tensor:
    c, pred, true = to_index(pred, true)

    def _apply(pred, true) -> torch.Tensor:
        true = true.view(-1)
        pred = pred.view(-1)
        tp, t, p = (
            x.bincount(minlength=c).clamp_(1).double()
            for x in (true[true == pred], true, pred))
        return 2 * tp / (t + p)

    if macro:
        return _apply(pred, true)

    b = pred.shape[0]
    out = map(_apply, pred.view(b, -1).unbind(), true.view(b, -1).unbind())
    return torch.mean(torch.stack([*out]), dim=0)
