import numpy as np
from matplotlib import pyplot as P

from glow.core import timer, sizeof
from glow.iters import mapped

_DEATH_RATE = 0


def source(size):
    return range(size)

    # deads = np.random.uniform(size=size)
    # # print(np.where(deads < _DEATH_RATE / size)[0].tolist())
    # for seed, death in enumerate(deads):
    #     if death < _DEATH_RATE / size:
    #         raise ValueError(f'Source died: {seed}')
    #     else:
    #         yield seed


def do_work(seed, offset):
    rng = np.random.RandomState(seed + offset)
    # n = rng.randint(10, 12)
    n = 5
    a = rng.rand(2**n, 2**n)
    b = rng.rand(2**n, 2**n)
    (a @ b).sum()
    if rng.uniform() < _DEATH_RATE / n:
        raise ValueError(f'Worker died: {seed}') from None
    return seed


class Worker:
    def __init__(self, n):
        self.array = np.random.rand(n)

    def __call__(self, *args):
        return do_work(*args)


if __name__ == '__main__':
    n = 1000
    order = 20
    stats = [0] * order
    sizes = [0] * order
    for m in range(order):
        worker = Worker(2 ** m)
        # worker = process
        res = mapped(
            worker,
            source(n),
            np.random.randint(2 ** 10, size=n),
            workers=8,
            offload=True,
        )
        with timer(m, stats) as time:
            assert list(res) == list(range(n))
        sizes[m] = sizeof(worker)

    P.plot(sizes, stats)
    P.loglog()
    P.show()
