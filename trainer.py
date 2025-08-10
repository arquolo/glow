from __future__ import annotations  # until 3.10

import sys
from collections import abc
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.cuda import amp
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


@dataclass
class Runner:
    best = float('+Inf')
    best_epoch = 0
    epoch = 0

    device: torch.device | str | None

    model: torch.nn.Module
    optim: torch.optim.Optimizer
    criterion: abc.Callable
    track: str
    logdir: (str | None) = None

    def load(self):
        if not (path := resolve_last_ckpt(self.logdir)):
            return
        for k, v in torch.load(path).items():
            if k in ('model', 'optim'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)

    def save(self):
        state = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'best': self.best,
            'best_epoch': self.best_epoch,
            'epoch': self.epoch,
        }
        torch.save(state, f'{self.checkpoints}/{self.epoch}.pth')

    def log_sota(self):
        title = (f'{Path(sys.argv[0]).name}: '
                 f'{self.best:.3f} SoTA (run {self.best_epoch}/{self.epoch})')
        print(f'\33]0;{title}\a', end='', flush=True)

    def do_step(self, data, label, is_train) -> dict:
        self.model.train(is_train)

        with torch.set_grad_enabled(is_train):
            if torch.cuda.device_count() > 1:
                # * Only forward pass will be speeded up this way
                out = data_parallel(self.model, data.to(self.device))
            else:
                out = self.model(data.to(self.device))
            loss = self.criterion(out, label.cuda())

        if is_train:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        mse = F.mse_loss(out.argmax(1).cpu(), label)
        return {'loss': loss.item(), 'mse': mse.item()}

    def run_loader(self,
                   cat: str,
                   loader: DataLoader,
                   writer: SummaryWriter,
                   is_train: bool = False):
        scores = {'loss': 0., 'mse': 0.}

        with tqdm(loader, leave=False, desc=cat) as bar:
            for step, (data, label) in enumerate(bar, 1):
                updates = self.do_step(data, label, is_train)
                for name, score in updates.items():
                    scores[name] += (score - scores[name]) / step
                bar.set_postfix(scores)

        scores['rmse'] = scores['mse'] ** 0.5
        line = f'{self.epoch:04d}/{cat}:' + ' '.join(
            f'{name}={score:.4f}' for name, score in scores.items())
        tqdm.write(line)
        for name, score in scores.items():
            writer.add_scalar(name, score, self.epoch)

        if not is_train:
            if scores[self.track] > self.best:
                self.best_epoch = self.epoch
                self.best = scores[self.track]
                self.save()
            self.log_sota()

    def run(self, loaders_and_writers: dict, epochs: int = 1):
        with ExitStack() as stack:
            for _, writer in loaders_and_writers.values():
                stack.enter_context(writer)

            for self.epoch in tqdm(range(self.epoch + 1, epochs + 1)):
                for cat, (loader, writer) in loaders_and_writers.items():
                    is_train = (cat == 'train')
                    self.run_loader(cat, loader, writer, is_train=is_train)


IScalars = abc.Mapping[str, float]
ITensors = abc.Mapping[str, np.ndarray]


class Scores(NamedTuple):
    scalars: dict[str, float]
    tensors: dict[str, np.ndarray]


class IMeter:
    def from_batch(self, out: torch.Tensor, target: torch.Tensor) -> Scores:
        raise NotImplementedError

    def from_epoch(self, batch_scores) -> Scores:
        raise NotImplementedError


class ILogger:
    def on_batch_end(self, scalars: IScalars) -> None:
        raise NotImplementedError

    def on_epoch_end(self, scalars: IScalars, tensors: ITensors) -> None:
        raise NotImplementedError


with self():  # setup writers, loggers
  for epoch in self.epochs:
    with epoch.ctx():
      for loader in loaders:
        with loader.ctx() as ctx:
          # .train / .set_grad

          for data in loader:
            with data.ctx() as ctx:
              ctx.do_step()
              # with torch.cuda.amp(fp16):
              #   out = model(input)
              # loss = get_loss(out, target)
              # if ctx.is_train:
              #   opt.zero_grad()
              #   loss.backward()
              #   opt.step()
              #   ctx.sched_step()
              # batch_scores = ctx.meter.from_batch(out, target)
                            #
                    # .collect_loader_scores
                    # .log_loader_scores
                    # .make_checkpoint
            # .log_
