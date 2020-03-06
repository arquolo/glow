from pprint import pprint

import glow
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

DEATH_RATE = 0
SIZE = 100


class Worker:
    def __init__(self, n):
        self.array = np.random.rand(int(n)).astype('f4')

    def __call__(self, _):
        return


def test_ipc_speed():
    order = 26
    hops = 5
    loops = 10
    stats, sizes = {}, {}
    for m in [order * hops - 1] + [*range(order * hops)[::-1]]:
        worker = Worker(2 ** (m / hops))
        sizes[m / hops] = glow.sizeof(worker)
        # with glow.timer(
        #         callback=lambda t: stats.__setitem__(m / hops, t / loops)):
        #     for _ in glow.mapped(
        #             worker, range(loops), workers=1, chunk_size=1):
        #         pass
        it = glow.mapped(worker, range(loops), workers=1, chunk_size=1)
        with glow.timer(
                callback=lambda t: stats.__setitem__(m / hops, t / loops)):
            for _ in it:
                pass

    if plt is None:
        pprint(stats)
        return
    print(
        'max bytes/s:',
        glow.Size((np.asarray([*sizes.values()]) /
                   np.asarray([*stats.values()])).max()))
    plt.figure(figsize=(4, 4))
    plt.plot([*sizes.values()],
             np.asarray([*sizes.values()]) / np.asarray([*stats.values()]))
    plt.ylim((1, 1e12))
    plt.ylabel('bytes/s')
    plt.xlabel('size')
    plt.loglog()
    plt.show()


def source(size):
    deads = np.random.uniform(size=size).astype('f4')
    print(np.where(deads < DEATH_RATE)[0].tolist()[:10])
    for seed, death in enumerate(deads):
        if death < DEATH_RATE:
            raise ValueError(f'Source died: {seed}')
        else:
            yield seed


def do_work(seed, offset):
    rng = np.random.RandomState(seed + offset)
    n = 10
    a = rng.rand(2 ** n, 2 ** n).astype('f4')
    b = rng.rand(2 ** n, 2 ** n).astype('f4')
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
    # sources = map(glow.buffered, sources)
    res = glow.mapped(do_work, *sources, chunk_size=1, ordered=False)
    print('start main', end='')
    for r in res:
        print(end=f'\rmain {r} computes...')
        rng = np.random.RandomState(r)
        n = 10
        a = rng.rand(2 ** n, 2 ** n).astype('f4')
        b = rng.rand(2 ** n, 2 ** n).astype('f4')
        (a @ b).sum()
        yield r
        print(end=f'\rmain {r} waits...')
    print('\rmain done')


def test_interrupt():
    rs = [*_test_interrupt()]
    assert {*rs} == {*range(SIZE)}
    # assert rs == [*range(SIZE)]


def test_interrupt_with_buffer():
    rs = [*glow.buffered(_test_interrupt())]
    assert {*rs} == {*range(SIZE)}
    # assert rs == [*range(SIZE)]


if __name__ == '__main__':
    # test_interrupt()
    # test_interrupt_with_buffer()
    test_ipc_speed()
