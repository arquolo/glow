import numpy as np
from matplotlib import pyplot as P

from glow import buffered, mapped, timer, sizeof

DEATH_RATE = 0
SIZE = 100


class Worker:
    def __init__(self, n):
        self.array = np.random.rand(n)

    def __call__(self, _):
        return


def test_ipc_speed():
    order = 25
    stats = [0] * order
    sizes = [0] * order
    for m in reversed(range(order)):
        worker = Worker(2 ** m)
        sizes[m] = sizeof(worker)
        with timer(m, stats):
            for _ in mapped(worker, range(10), workers=1, offload=True):
                pass

    P.figure(figsize=(4, 4))
    P.plot(sizes, stats)
    P.ylim((0.001, 10))
    P.ylabel('time')
    P.xlabel('size of worker')
    P.loglog()
    P.show()


def source(size):
    deads = np.random.uniform(size=size)
    print(np.where(deads < DEATH_RATE)[0].tolist()[:10])
    for seed, death in enumerate(deads):
        if death < DEATH_RATE:
            raise ValueError(f'Source died: {seed}')
        else:
            yield seed


def do_work(seed, offset):
    rng = np.random.RandomState(seed + offset)
    n = 10
    a = rng.rand(2 ** n, 2 ** n)
    b = rng.rand(2 ** n, 2 ** n)
    (a @ b).sum()
    if rng.uniform() < DEATH_RATE:
        raise ValueError(f'Worker died: {seed}') from None
    return seed


def _test_interrupt():
    """Should die gracefully on Ctrl-C"""
    sources = (
        source(SIZE),
        np.random.randint(2 ** 10, size=SIZE),
    )
    res = mapped(do_work, *map(buffered, sources), offload=True)
    print('start main', end='')
    for r in res:
        print(end=f'\rmain {r} computes...')
        rng = np.random.RandomState(r)
        n = 10
        a = rng.rand(2 ** n, 2 ** n)
        b = rng.rand(2 ** n, 2 ** n)
        (a @ b).sum()
        yield r
        print(end=f'\rmain {r} waits...')
    print('\rmain done')


def test_interrupt():
    ls = _test_interrupt()
    assert list(ls) == list(range(SIZE))


def test_interrupt_with_buffer():
    ls = buffered(_test_interrupt())
    assert list(ls) == list(range(SIZE))


if __name__ == '__main__':
    test_interrupt()
    test_interrupt_with_buffer()
    test_ipc_speed()
