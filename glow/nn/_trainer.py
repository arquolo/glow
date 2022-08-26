from __future__ import annotations

__all__ = ['Trainer']

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim
from tqdm.auto import tqdm

from .. import ichunked
from .. import metrics as m
from ._loader import _Loader
from .amp import Grads, get_grads
from .util import eval_


class Stage:
    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        raise NotImplementedError


@dataclass(frozen=True)
class EvalStage(Stage):
    net: nn.Module
    device: torch.device
    fp16: bool

    def _infer(self, data: torch.Tensor,
               target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.autocast(self.device.type, enabled=self.fp16):
            out = self.net(data.to(self.device, non_blocking=True))

        return out, target

    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        with eval_(self.net), torch.inference_mode():
            for data, target in loader:
                yield self._infer(
                    data.to(self.device, non_blocking=True),
                    target.to(self.device, non_blocking=True),
                )


@dataclass(frozen=True)
class TrainStage(Stage):
    net: nn.Module
    device: torch.device
    fp16: bool
    criterion: Callable[..., torch.Tensor]
    grads: Grads
    grad_steps: int

    def _step(self, data: torch.Tensor,
              target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.autocast(self.device.type, enabled=self.fp16):
            out = self.net(data.to(self.device, non_blocking=True))
            loss = self.criterion(out, target)

        self.grads.backward(loss)
        return out.detach(), target

    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        for batches in ichunked(loader, self.grad_steps):
            with self.grads:
                for data, target in batches:
                    yield self._step(
                        data.to(self.device, non_blocking=True),
                        target.to(self.device, non_blocking=True),
                    )
                # Clip norm here if needed


class Trainer:
    def __init__(self,
                 net: nn.Module,
                 opt: torch.optim.Optimizer,
                 criterion: Callable[..., torch.Tensor],
                 metrics: Iterable[m.Metric],
                 device: torch.device,
                 sched: torch.optim.lr_scheduler._LRScheduler | None = None,
                 fp16: bool = False,
                 grad_steps: int = 1) -> None:
        self.metrics = [*metrics]
        grads = get_grads(opt, sched, fp16=fp16, max_retries=0)
        self.stages = (
            TrainStage(net, device, fp16, criterion, grads, grad_steps),
            EvalStage(net, device, fp16),
        )

    def _run(self, stage: Stage, loader: _Loader, pbar: tqdm) -> m.Scores:
        meter = m.compose(*self.metrics)
        scores = m.Scores()

        for out in stage(loader):
            scores = meter.send(out)
            pbar.set_postfix(scores.scalars)
            pbar.update()

        return scores

    def train(self, loader: _Loader, pbar: tqdm) -> m.Scores:
        return self._run(self.stages[0], loader, pbar)

    def eval(self, loader: _Loader, pbar: tqdm) -> m.Scores:
        return self._run(self.stages[1], loader, pbar)

    def run(self,
            train_loader: _Loader,
            eval_loader: _Loader,
            epochs: int = 1):
        for i in tqdm(range(1, 1 + epochs), smoothing=0):
            with tqdm(train_loader, desc='train', leave=False) as bar:
                tscalars = self.train(bar, bar).scalars

            with tqdm(eval_loader, desc='val', leave=False) as bar:
                vscalars = self.eval(bar, bar).scalars

            assert tscalars.keys() == vscalars.keys()
            tags = sorted(tscalars.keys() | vscalars.keys())

            # TODO: those lines should be moved outsize into loggers
            line = ','.join(
                f'{tag}: ' + '/'.join(f'{s[tag]:.3f}'
                                      for s in (tscalars, vscalars))
                for tag in tags)
            print(f'[{i:03d}] {line}')


# TODO: define Logger/Handler classes
# TODO:  i.e. Logger which handles tensors and scalars
# TODO:  and Handler flavours like StdoutHandler, TensorBoardHandler, etc.
