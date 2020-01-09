import abc
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Generator, Tuple

import torch
from torch import LongTensor, Tensor

_EPS = torch.finfo(torch.float).eps  # type: ignore
_MetricFn = Callable[..., Tensor]


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, pred, true) -> Tensor:
        raise NotImplementedError

    def collect(self, state) -> Dict[str, Tensor]:
        raise state


@dataclass
class MetricFn(Metric):
    fn: _MetricFn

    def __call__(self, pred, true) -> Tensor:
        return self.fn(pred, true)

    def collect(self, state):
        return {self.fn.__name__: state}


def _to_index(pred, true) -> Tuple[int, LongTensor, LongTensor]:
    c = pred.shape[1]
    pred = pred.argmax(dim=1)

    if true.min() < 0 or true.max() >= c:
        mask = (true >= 0) & (true < c)
        true = true[mask][None]
        pred = pred[mask][None]

    return c, pred, true


def _to_prob(pred, true) -> Tuple[int, LongTensor, Tensor]:
    b, c = pred.shape[:2]
    pred = pred.softmax(dim=1)

    if true.min() < 0 or true.max() >= c:
        true = true.view(-1)
        pred = pred.view(b, c, -1).permute(0, 2, 1).view(-1, c)

        mask = (true >= 0) & (true < c)
        true = true[mask]
        pred = pred[mask]

    return c, pred, true


def _running_mean(fn: Metric
                  ) -> Generator[Dict[str, Tensor], Tuple[Tensor, ...], None]:
    assert isinstance(fn, Metric)

    args = yield {}
    state = torch.as_tensor(fn(*args))
    for step in itertools.count(2):
        args = yield fn.collect(state)
        state += (torch.as_tensor(fn(*args)) - state) / step


# ---------------- standalone metrics ----------------

def accuracy_(pred, true):
    _, pred, true = _to_index(pred, true)
    return (true == pred).double().mean()


def dice(pred, true, macro=False):
    c, pred, true = _to_index(pred, true)

    def _apply(pred, true):
        true = true.view(-1)
        pred = pred.view(-1)
        tp, t, p = (
            x.bincount(minlength=c).clamp_(1).double()
            for x in (true[true == pred], true, pred)
        )
        return 2 * tp / (t + p)

    if macro:
        return _apply(pred, true)

    b = pred.shape[0]
    true = true.view(b, -1)
    pred = pred.view(b, -1)
    return torch.mean(list(map(_apply, pred, true)), dim=0)


# ---------------- cm-based metrics ----------------

class MultiMetric(Metric):
    def __init__(self, **callables):
        self.callables = callables

    def collect(self, state):
        return {key: fn(state) for key, fn in self.callables.items()}


class Confusion(MultiMetric):
    def __call__(self, pred, true) -> Tensor:
        c, pred, true = _to_index(pred, true)
        true = true.view(-1)
        pred = pred.view(-1)

        # v1. Inspired by (kornia)[github.com/kornia/kornia]
        # mat = pred.add(c, true).bincount(minlength=c**2).view(c, c)

        # v2. Faster
        mat = torch.zeros(c, c, dtype=torch.long)
        mat = mat.index_put_((true, pred), torch.tensor(1), accumulate=True)

        return mat.double() / mat.sum()

    def collect(self, mat):
        c = mat.shape[0]
        return {f'cm{c}': mat, **super().collect(mat)}


class ConfusionGrad(Confusion):
    def __call__(self, pred, true):
        c, pred, true = _to_prob(pred, true)
        true = true.view(-1)
        pred = pred.view(-1, c)
        mat = torch.zeros(c, c, dtype=pred.dtype).index_add(0, true, pred)
        return mat.double() / mat.sum()


def accuracy(mat):
    return mat.diag().sum() / mat.sum().clamp(_EPS)


def accuracy_balanced(mat):
    return (mat.diag() / mat.sum(1).clamp(_EPS)).mean()


def kappa(mat):
    expected = mat.sum(0) @ mat.sum(1)
    observed = mat.diag().sum()
    return (observed - expected) / (1 - expected).clamp(_EPS)


def iou(mat):
    return mat.diag() / (mat.sum(0) + mat.sum(1) - mat.diag()).clamp(_EPS)


# ---------------- main ----------------

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torch.nn import Module, Parameter
    from torch.optim import AdamW  # type: ignore
    from tqdm.auto import tqdm

    metrics: Tuple[Metric, ...] = (
        MetricFn(accuracy_),
        Confusion(
            acc=accuracy,
            accb=accuracy_balanced,
            iou=iou,
            kappa=kappa,
        ),
    )

    c = 8
    b = 128
    true = torch.randint(c, size=[b])
    pred = torch.randn(b, c, requires_grad=True)

    class Model(Module):
        def forward(self, x):
            return md.param

    md = Model()
    md.param = Parameter(pred)  # type: ignore

    optim = AdamW(md.parameters())
    cmg = ConfusionGrad()

    plt.ion()
    _, ax = plt.subplots(ncols=4)
    ax[2].plot(true.numpy())
    for _ in tqdm(range(100)):
        for _ in range(10):
            md.zero_grad()
            cm = cmg(md(None), true)
            # loss = -accuracy_balanced(cm)
            # loss = -kappa(cm)
            loss = -accuracy(cm)
            loss.backward()
            optim.step()
            del loss
            cm = cm.detach()

        ax[0].imshow(pred.detach().numpy())
        ax[1].imshow(pred.detach().softmax(1).numpy())
        ax[2].cla()
        ax[2].plot(true.numpy())
        ax[2].plot(pred.detach().argmax(1).numpy())
        ax[3].imshow(cm.detach().numpy())

        plt.pause(1e-2)

    with torch.no_grad():
        updates = [_running_mean(fn) for fn in metrics]
        for u in updates:
            next(u)
            d = u.send((pred, true))
            print({
                k: v.mul(10 ** 3).round().div(10 ** 3).tolist()  # type: ignore
                for k, v in d.items()
            })
