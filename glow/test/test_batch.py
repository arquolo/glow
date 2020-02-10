import random

import glow
import pytest
from unittest.mock import MagicMock


@pytest.mark.parametrize('deco', [glow.batched, glow.batched_async])
def test_as_is(deco):
    load = MagicMock(side_effect=lambda x: x)

    @deco
    def load_batch(xs):
        return [load(x) for x in xs]

    for _ in range(5):
        numbers = random.choices(range(20), k=5)
        results = load_batch(numbers)
        assert numbers == results

    called_with = (x for (x, ), _ in load.call_args_list)
    assert sorted(called_with) == sorted(load_batch.cache)


@pytest.mark.parametrize('workers', [2, 4] * 10)
@pytest.mark.parametrize('deco', [glow.batched, glow.batched_async])
def test_thread_safe(deco, workers):
    @deco
    def load_batch(xs):
        return [*xs]

    numbers = random.choices(range(100), k=100)
    nchunks = glow.chunked(numbers, 5)
    rchunks = glow.mapped(load_batch, nchunks, workers=workers)
    assert [r for c in rchunks for r in c] == numbers
