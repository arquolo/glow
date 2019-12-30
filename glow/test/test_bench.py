from itertools import islice

from tqdm.auto import tqdm
from glow.core import time_this


def iter_fibs():
    prev, cur = 0, 1
    while True:
        prev, cur = cur, prev + cur
        yield cur


@time_this
def fibs(n):
    gen = islice(iter_fibs(), n)
    gen = tqdm(gen)
    return max(x.bit_length() for x in gen)


def test_fibs():
    assert fibs(100000) == 69424
