import argparse
import pathlib
from typing import DefaultDict, List

import glow
import glow.nn as gnn
import glow.metrics as m
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms as tfs
from tqdm.auto import tqdm


def make_model_default():
    return nn.Sequential(
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
                skip=0.1))

    return nn.Sequential(
        # > 32^2
        # nn.ZeroPad2d(2),
        # > 16^2
        # conv_down(1, init),
        # > 16^2
        conv_down(3, init),
        # > 8^2
        conv_down(init, init * 2),
        # > 4^2
        conv_down(init * 2, init * 4),
        conv(init * 4, init * 8, pad=2),
        # > 1
        nn.AdaptiveAvgPool2d(1),
        gnn.View(-1),
        nn.Linear(init * 8, 10),
    )


def loop(loader, step_fn, metrics):
    meter = m.compose(*metrics)
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        out = step_fn(data, target)
        scores = meter.send((out, target))
        yield {k: v.item() for k, v in scores.items() if v.numel() == 1}


def main(root: pathlib.Path, batch_size: int, width: int, epochs: int):
    tft = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tfv = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    path = root / 'cifar10'
    tset = datasets.CIFAR10(path, transform=tft, train=True, download=True)
    vset = datasets.CIFAR10(path, transform=tfv, train=False)
    """
    tf = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5,), (0.5,)),
    ])
    tset = datasets.MNIST(root, transform=tf, train=True, download=True)
    vset = datasets.MNIST(root, transform=tf, train=False)
    """
    @glow.repeatable(hint=tset.__len__)
    def sampler():
        return torch.randperm(len(tset))

    tload = gnn.make_loader(
        tset, sampler(), batch_size=batch_size, multiprocessing=False)
    vload = gnn.make_loader(vset, batch_size=200, multiprocessing=False)

    # net = make_model_default()
    net = make_model_new(init=width)
    net.cuda()
    print('params:', gnn.param_count(net))

    optim = gnn.RAdam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    metrics = [
        m.Lambda(loss_fn, name='loss'),
        m.Confusion(acc=m.accuracy, kappa=m.kappa),
    ]

    def train_step(data, target):
        net.train()
        optim.zero_grad()
        out = net(data)
        loss_fn(out, target).backward()
        optim.step()
        return out.detach()

    def val_step(data, target):
        net.eval()
        with torch.no_grad():
            return net(data)

    scores = DefaultDict[str, List](list)
    for _ in tqdm(range(epochs), desc='epochs'):
        for split in glow.ichunked(
                tqdm(tload, desc='train', leave=False),
                size=8000 // batch_size):
            *_, mt = loop(split, train_step, metrics)
            *_, mv = loop(
                tqdm(vload, desc='val', leave=False), val_step, metrics)

            names = sorted({*mt} & {*mv})
            # print(', '.join(f'{k}: {mt[k]:.3f}/{mv[k]:.3f}' for k in names))
            for name in names:
                scores[name].append([mt[name], mv[name]])

    _, axes = plt.subplots(ncols=len(scores))
    for (title, data), ax in zip(scores.items(), axes):
        ax.legend(ax.plot(data), ['train', 'val'])
        ax.set_title(title)
    plt.show()
    return [max(ks) for ks in zip(*scores['kappa'])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=pathlib.Path, help='location to store dataset')
    parser.add_argument(
        '--batch-size', type=int, default=4, help='batch size for train')
    parser.add_argument('--epochs', type=int, default=2, help='epochs')
    parser.add_argument(
        '--width', type=int, default=32, help='width of network')
    args = parser.parse_args()
    main(**vars(args))
