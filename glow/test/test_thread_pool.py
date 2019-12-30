import numpy as np
from glow import buffered, mapped, timer, sizeof
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

DEATH_RATE = 0
SIZE = 100


class Worker:
    def __init__(self, n):
        self.array = np.random.rand(int(n))

    def __call__(self, _):
        return


def test_ipc_speed():
    order = 25
    hops = 5
    stats, sizes = {}, {}
    for m in [order * hops - 1] + list(range(order * hops))[::-1]:
        worker = Worker(2 ** (m / hops))
        sizes[m / hops] = sizeof(worker)
        it = mapped(worker, range(10), workers=1, chunk_size=1)
        with timer(m / hops, stats):
            for _ in it:
                pass

    if plt is None:
        return
    plt.figure(figsize=(4, 4))
    plt.plot(list(sizes.values()), list(stats.values()))
    plt.ylim((0.001, 10))
    plt.ylabel('time')
    plt.xlabel('size of worker')
    plt.loglog()
    plt.show()


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
    # sources = map(buffered, sources)
    res = mapped(do_work, *sources, chunk_size=1, ordered=False)
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
    rs = list(ls)
    assert set(rs) == set(range(SIZE))
    # assert rs == list(range(SIZE))


def test_interrupt_with_buffer():
    ls = buffered(_test_interrupt())
    rs = list(ls)
    assert set(rs) == set(range(SIZE))
    # assert rs == list(range(SIZE))


if __name__ == '__main__':
    # test_interrupt()
    # test_interrupt_with_buffer()
    test_ipc_speed()
