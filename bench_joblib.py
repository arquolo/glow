import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import glow
import glow.joblib as jl


N_JOBS = 24
MAX_ITERS = 256
MAX_TRANSFER = 2 ** 32  # 4 GB
RANK = 100
# SAME = False
SAME = True


def _reduce(a):
    return np.mean(a)


def _mkarr(seed):
    return np.random.default_rng(seed).random(PACK // 8)


def iter_jobs(seeds):
    return [_mkarr(seeds[0])] * len(seeds) if SAME else map(_mkarr, seeds)


def bench_glow(seeds):
    return glow.mapped(
        _reduce, iter_jobs(seeds), chunk_size=4, latency=12, workers=N_JOBS)


def bench_joblib_loky(seeds):
    return jl.Parallel(
        n_jobs=N_JOBS, max_nbytes=NBYTES)(
            jl.delayed(_reduce)(a) for a in iter_jobs(seeds))


def bench_joblib_mp(seeds):
    return jl.Parallel(
        n_jobs=N_JOBS, max_nbytes=NBYTES, backend='multiprocessing')(
            jl.delayed(_reduce)(a) for a in iter_jobs(seeds))


if __name__ == '__main__':
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(
        111, xlabel='size', ylabel='time', xscale='log', yscale='log')

    for fn, NBYTES in (
        (bench_glow, None),
        (bench_joblib_loky, None),
        (bench_joblib_loky, 0),
        (bench_joblib_mp, None),
        (bench_joblib_mp, 0),
    ):
        label = fn.__name__
        if NBYTES is not None:
            label += '-mmap'

        times: dict = {}
        for PACK in tqdm(np.logspace(25, 5, num=RANK, base=2, dtype=int),
                         desc=label):
            loops = min(MAX_TRANSFER // PACK, MAX_ITERS)
            seeds = np.random.randint(2 ** 20, size=loops)

            rs = [*map(_reduce, iter_jobs(seeds))]
            with glow.timer(
                    callback=lambda t: times.__setitem__(PACK, t / loops)):
                ps = fn(seeds)
                # ps = tqdm(ps, total=loops, desc=f'{glow.Si(PACK)}B',
                #           leave=False)
                assert rs == [*ps]

        ax.plot([*times.keys()], [*times.values()], label=label)

    fig.legend()
    plt.show()
