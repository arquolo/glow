import time

from glow import map_n

N = 40
NW = 4
FN_DIE = N
GEN_DIE = N
PREFETCH = None
ORDER = True
TSTEP = 0.02

NCOLS = 100

s: dict[int, list] = {}


def gen():
    for x in range(N):
        if x == GEN_DIE:
            raise ValueError
        s.setdefault(x, []).append(time.perf_counter())
        yield x


def fn(x):
    s[x].append(time.perf_counter())
    time.sleep(0.1 * (1 - x / N))  # emulate GIL-unrestricted work
    if x == FN_DIE:
        raise ValueError
    s[x].append(time.perf_counter())
    return x


init = time.perf_counter()
for i, x in enumerate(
        map_n(
            fn,
            gen(),
            max_workers=NW,
            prefetch=PREFETCH,
            order=ORDER,
        ), 1):
    s[x].append(time.perf_counter())
    wt = i * TSTEP - (time.perf_counter() - init)  # emulate slow main
    if wt > 0:
        time.sleep(wt)

s = {x: ts for x, ts in s.items() if len(ts) == 4}

low = min(sum(s.values(), []))
s = {x: [t - low for t in ts] for x, ts in s.items()}

avg_done = sum(t2 for _, _, t2, _ in s.values()) / len(s)
all_done = max(t2 for _, _, t2, _ in s.values())
avg_return = sum(t3 for _, _, _, t3 in s.values()) / len(s)
all_return = max(t3 for _, _, _, t3 in s.values())

step = all_return / NCOLS
for x, ts in s.items():
    t0, t1, t2, t3 = ts = [round(t / step) for t in ts]
    print(f'{x:03d}:',
          ' ' * t0 + '-' * (t1 - t0) + '#' * (t2 - t1) + '-' * (t3 - t2))

print(f'{avg_done = :.3f}s, {avg_return = :.3f}s')
print(f'{all_done = :.3f}s, {all_return = :.3f}s')
print('results =', ', '.join(
    f'{x}' for x in sorted(s, key=lambda x: s[x][-1])))
