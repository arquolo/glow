import argparse
import contextlib
import pathlib
import random
from typing import DefaultDict

import glow
import glow.metrics as m
import glow.nn as gnn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

DEVICE = torch.device('cuda')

torch.backends.cudnn.benchmark = True  # type: ignore
glow.lock_seed(42)


def train(net: nn.Module, loader, optim, criterion):
    for input_, target in loader:
        target = target.to(DEVICE)
        with optim:
            out = net(input_)
            loss = criterion(out, target)
            optim.backward(loss)

        yield (out.detach(), target)


def validate(net: nn.Module, loader):
    with contextlib.ExitStack() as stack:
        stack.callback(net.train, net.training)
        net.eval()
        stack.enter_context(torch.no_grad())
        for input_, target in loader:
            target = target.to(DEVICE)
            out = net(input_)
            yield (out, target)


# ------------------------------ define model ------------------------------


def make_model_default():
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(3, 6, 5),  # > 28^2
        nn.ReLU(),
        nn.MaxPool2d(2),  # > 14^2
        nn.Conv2d(6, 16, 5),  # > 10^2
        nn.ReLU(),
        nn.MaxPool2d(2),  # > 5^2
        gnn.View(-1),  # > 1:400
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


def make_model_new(init=16):
    def conv(cin, cout=None, groups=1, pad=2, stride=1):
        cout = cout or cin
        ksize = stride + pad * 2
        return nn.Sequential(
            nn.Conv2d(
                cin, cout, ksize, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def conv_down(cin, cout=None):
        cout = cout or cin
        return nn.Sequential(
            conv(cin, cout, pad=1, stride=2),
            gnn.Sum(
                conv(cout, cout * 2),
                conv(cout * 2, pad=2, groups=cout * 2),
                conv(cout * 2, cout)[:-1],
                tail=nn.ReLU(),
                skip=0.1),
        )

    return nn.Sequential(
        conv_down(3, init),  # > 16^2
        conv_down(init, init * 2),  # > 8^2
        conv_down(init * 2, init * 4),  # > 4^2
        conv(init * 4, init * 8, pad=2),
        nn.AdaptiveAvgPool2d(1),  # > 1
        gnn.View(-1),
        nn.Linear(init * 8, 10),
    )


# parse args

parser = argparse.ArgumentParser()
parser.add_argument(
    'root', type=pathlib.Path, help='location to store dataset')
parser.add_argument('--batch-size', type=int, default=4, help='train batch')
parser.add_argument('--epochs', type=int, default=12, help='count of epochs')
parser.add_argument('--steps-per-epoch', type=int, help='steps per epoch')
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument(
    '--fp16', action='store_true', help='enable mixed precision mode')
parser.add_argument('--plot', action='store_true', help='disable plot')

args = parser.parse_args()

epoch_len = (8000 // args.batch_size
             if args.steps_per_epoch is None else args.steps_per_epoch)
sample_size = args.epochs * epoch_len * args.batch_size

ds = CIFAR10(args.root / 'cifar10', transform=ToTensor(), download=True)
ds_val = CIFAR10(args.root / 'cifar10', transform=ToTensor(), train=False)


@glow.repeatable(hint=lambda: sample_size)
def sampler(rg=np.random.default_rng()):
    return rg.integers(len(ds), size=sample_size)


loader = gnn.make_loader(
    ds, sampler(), batch_size=args.batch_size, multiprocessing=False)
val_loader = gnn.make_loader(ds_val, batch_size=100, multiprocessing=False)


def collect(metrics, pbar, outputs):
    scores = {}
    meter = m.compose(*metrics)

    for (out, target) in outputs:
        scores = meter.send((out, target))
        scores = {k: v.item() for k, v in scores.items() if v.numel() == 1}
        pbar.set_postfix(scores)
        pbar.update()

    return scores


# net = make_model_default()
net = make_model_new(args.width)
print(gnn.param_count(net))

_optim = torch.optim.AdamW(net.parameters())
optim = gnn.amp_init_opt(
    net, _optim, device=DEVICE, fp16=args.fp16, allow_skip=True)

criterion = nn.CrossEntropyLoss()
metrics = [
    m.Lambda(criterion, name='loss'),
    m.Confusion(acc=m.accuracy, kappa=m.kappa),
]

history = DefaultDict[str, list](list)

with tqdm(total=epoch_len * args.epochs, desc='train') as pbar:
    for t_out in glow.ichunked(
            train(net, loader, optim, criterion), epoch_len):
        with tqdm(total=len(val_loader), desc='val', leave=False) as pbar_val:
            _scores = (
                collect(metrics, pbar, t_out),
                collect(metrics, pbar_val, validate(net, val_loader)),
            )
            _tags = sorted({tag for s in _scores for tag in s})
            scores = {tag: [s[tag] for s in _scores] for tag in _tags}

        for tag, values in scores.items():
            history[tag].append(values)
        print(', '.join(f'{tag}: {{:.3f}}/{{:.3f}}'.format(*values)
                        for tag, values in scores.items()))

if args.plot:
    fig, axes = plt.subplots(ncols=len(history))
    fig.suptitle(f'batch_size={args.batch_size}, fp16={args.fp16}')
    for ax, (tag, values) in zip(axes, history.items()):
        ax.legend(ax.plot(values), ['train', 'val'])
        ax.set_title(tag)
        ax.set_ylim([
            int(min(x for xs in values for x in xs)),
            int(max(x for xs in values for x in xs) + 0.999)
        ])
    plt.show()
