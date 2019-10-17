__all__ = ('tiramisu', )

import itertools

from torch import nn

from ..iters import windowed
from . import Cat, DenseBlock, conv, param_count


def tiramisu(cin, cout, init=48, depths=(4, 4), step=12):
    steps = itertools.accumulate([init] + [step * depth for depth in depths])
    steps = list(steps)

    core = [DenseBlock(steps[-2], depth=depths[-1], step=step)]
    slides = windowed(steps, size=3)
    for i, (f_in, f_middle, f_out) in reversed(list(enumerate(slides))):
        f_delta = f_out - f_middle
        depth = (f_middle - f_in) // step
        is_last = not i
        core = [
            DenseBlock(f_in, depth=depth, step=step, full=True),
            Cat(nn.Sequential(
                conv(f_middle, padding=0, order='NA-'),
                nn.MaxPool2d(2),
                *core,
                nn.ConvTranspose2d(f_delta, f_delta, 2, stride=2),
            )),
            DenseBlock(f_out, depth=depth, step=step, full=is_last),
        ]

    model = nn.Sequential(
        nn.Conv2d(cin, init, 3, padding=1),
        *core,
        nn.Conv2d(f_out + depth * step, cout, 1),
    )
    layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    params = param_count(model)
    model.name = f'tiramisu_{layers}_{params}'
    return model
