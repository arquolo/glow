import argparse
import pathlib
from typing import DefaultDict, List

import glow.nn
import glow.metrics as m
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms as tfs
from tqdm.auto import tqdm


def make_model_default():
    return nn.Sequential(
        nn.Conv2d(3, 6, 5),     # > 28^2
        nn.ReLU(),
        nn.MaxPool2d(2),        # > 14^2
        nn.Conv2d(6, 16, 5),    # > 10^2
        nn.ReLU(),
        nn.MaxPool2d(2),        # > 5^2
        nn.Conv2d(16, 120, 5),  # > 1
        nn.ReLU(),
        nn.Conv2d(120, 84, 1),
        nn.ReLU(),
        nn.Conv2d(84, 10, 1),
    )


def make_model_new(init=16):
    def conv(cin, cout=None, groups=1, pad=2, stride=1):
        cout = cout or cin
        ksize = stride + pad * 2
        return nn.Sequential(
            nn.Conv2d(cin, cout, ksize, stride,
                      padding=pad, bias=False, groups=groups),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def conv_down(cin, cout=None):
        cout = cout or cin
        return nn.Sequential(
            conv(cin, cout, pad=1, stride=2),
            glow.nn.Sum(
                conv(cout, cout * 2),
                conv(cout * 2, pad=2, groups=cout * 2),
                conv(cout * 2, cout)[:-1], tail=nn.ReLU(), skip=0.1
            ),
        )

    return nn.Sequential(
        conv_down(3, init),             # > 16^2
        conv_down(init, init * 2),      # > 8^2
        conv_down(init * 2, init * 4),  # > 4^2
        conv(init * 4, init * 8, pad=2),
        nn.AvgPool2d(4),                # > 1
        nn.Conv2d(init * 8, 10, 1),
    )


def loop(loader, step_fn, metrics, name=None):
    updates = [m.batch_averaged(f) for f in metrics]
    r = {}

    for data, target in tqdm(loader, desc=name, leave=False):
        data, target = data.cuda(), target[:, None, None].cuda()
        out = step_fn(data, target)
        with torch.no_grad():
            r.update({
                k: v.item() for u in updates
                for k, v in u.send((out, target)).items() if v.numel() == 1
            })
    return r


def main(root: pathlib.Path, batch_size: int):
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

    @glow.repeatable(hint=tset.__len__)
    def sampler():
        return torch.randperm(len(tset))

    tload = glow.nn.make_loader(tset, sampler(),
                                batch_size=batch_size, chunk_size=0)
    vload = glow.nn.make_loader(vset, batch_size=200, chunk_size=0)

    # net = make_model_default()
    net = make_model_new()
    net.cuda()
    print('params:', glow.Size(sum(p.numel() for p in net.parameters())))

    optim = glow.nn.RAdam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # def loss_fn(y_pred, y_true):
    #     labels = torch.zeros_like(y_pred)
    #     labels.scatter_(1, y_true[:, None, ...], 1)
    #     return torch.nn.BCEWithLogitsLoss()(y_pred, labels)

    metrics = [
        m.Lambda(loss_fn, name='loss'),
        m.Confusion(acc=m.accuracy, kappa=m.kappa),
    ]

    def train_step(data, target):
        net.train()
        optim.zero_grad()
        out = net(data)
        loss = loss_fn(out, target)
        loss.backward()
        optim.step()
        return out.detach()

    def val_step(data, target):
        net.eval()
        with torch.no_grad():
            return net(data)

    scores = DefaultDict[str, List](list)
    for _ in tqdm(range(20), desc='epochs'):
        splits = glow.ichunked(tload, 8000 // batch_size)
        for split in tqdm(splits, desc='sub-epochs', leave=False):
            mt_, mv_ = (
                loop(split, train_step, metrics, name='train'),
                loop(vload, val_step, metrics, name='val'),
            )
            names = sorted({*mt_} & {*mv_})
            print(', '.join(f'{k}: {mt_[k]:.3f}/{mv_[k]:.3f}' for k in names))
            for name in names:
                scores[name].append([mt_[name], mv_[name]])

    _, axes = plt.subplots(ncols=len(scores))
    for (title, data), ax in zip(scores.items(), axes):
        ax.legend(ax.plot(data), ['train', 'val'])
        ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=pathlib.Path, help='location to store dataset')
    parser.add_argument(
        '--batch-size', type=int, default=4, help='batch size for train')
    args = parser.parse_args()
    main(**vars(args))
