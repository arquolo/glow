import numpy as np
from matplotlib import pyplot as P

from glow.core import timer, sizeof
from glow.iters import mapped

DEATH_RATE = 0
SIZE = 1000


def source():
    deads = np.random.uniform(size=SIZE)
    print(np.where(deads < DEATH_RATE)[0].tolist()[:10])
    for seed, death in enumerate(deads):
        if death < DEATH_RATE:
            raise ValueError(f'Source died: {seed}')
        else:
            yield seed


def do_work(seed, offset):
    rng = np.random.RandomState(seed + offset)
    n = rng.randint(10, 12)
    # n = 5
    a = rng.rand(2**n, 2**n)
    b = rng.rand(2 ** n, 2 ** n)
    (a @ b).sum()
    if rng.uniform() < DEATH_RATE:
        raise ValueError(f'Worker died: {seed}') from None
    return seed


class Worker:
    def __init__(self, n):
        self.array = np.random.rand(n)

    def __call__(self, *args):
        return do_work(*args)


if __name__ == '__main__':
    order = 20
    stats = [0] * order
    sizes = [0] * order
    for m in range(order):
        # worker = Worker(2 ** m)
        worker = do_work
        res = mapped(
            worker,
            source(),
            np.random.randint(2 ** 10, size=SIZE),
            workers=8,
            offload=True,
        )
        with timer(m, stats) as time:
            ls = []
            print('start main', end='')
            for r in res:
                print(f'\rmain {r} computes...', end='')
                rng = np.random.RandomState(r)
                n = 12
                a = rng.rand(2 ** n, 2 ** n)
                b = rng.rand(2 ** n, 2 ** n)
                (a @ b).sum()
                ls.append(r)
                print(f'\rmain {r} waits...', end='')
            print('\rmain done')
            assert ls == list(range(SIZE))
        sizes[m] = sizeof(worker)

    P.plot(sizes, stats)
    P.loglog()
    P.show()
