"""Run with PYTHONPATH=src: python benchmarks/bench_shared_callable.py.

Compare repeated serialization with shared callable buffers. Timings include
proxy creation and result consumption, but exclude process startup. No timing
thresholds are asserted. Try several --mib, --tasks and --chunksize values.
"""

import argparse
from statistics import median
from time import perf_counter

import numpy as np
from loky import get_reusable_executor

from glow import get_executor, map_n


class Lookup:
    def __init__(self, size):
        self.values = np.arange(size, dtype=np.int64)

    def __call__(self, index):
        return int(self.values[index % self.values.size])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--tasks', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunksize', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if (
        min(*args.mib, args.tasks, args.workers, args.chunksize, args.repeat)
        < 1
    ):
        parser.error('all arguments must be positive')

    try:
        with get_executor(args.workers, mp=True) as executor:
            warmup = [executor.submit(pow, 2, 10) for _ in range(args.workers)]
            assert all(f.result() == 1024 for f in warmup)
        print('MiB  mode     median_s  speedup')
        for mib in args.mib:
            lookup = Lookup(mib * 1024 * 1024 // 8)
            samples = {mode: [] for mode in (True, 'mmap', 'shmem')}
            for repeat in range(args.repeat):
                modes = list(samples)
                # Rotate order to reduce systematic warm-cache bias.
                modes = modes[repeat % 3 :] + modes[: repeat % 3]
                for mode in modes:
                    start = perf_counter()
                    results = list(
                        map_n(
                            lookup,
                            range(args.tasks),
                            mp=mode,
                            max_workers=args.workers,
                            chunksize=args.chunksize,
                        )
                    )
                    samples[mode].append(perf_counter() - start)
                    assert results == [lookup(i) for i in range(args.tasks)]
            baseline = median(samples[True])
            for mode, times in samples.items():
                duration = median(times)
                print(
                    f'{mib:3}  {mode!s:7}  {duration:8.4f}  '
                    f'{baseline / duration:6.2f}x'
                )
    finally:
        get_reusable_executor().shutdown(wait=True)


if __name__ == '__main__':
    main()
