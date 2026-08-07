import asyncio
from asyncio import CancelledError

import pytest

from glow import MulticastQueue
from glow._mcqueue import QueueShutdownError


async def worker(mq: MulticastQueue[int], total: int) -> None:
    with mq:
        for x in range(total):
            await mq.put(x)


@pytest.mark.asyncio
async def test_seq() -> None:
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    seq = [x async for x in mq]
    assert seq == [0, 1]

    seq = [x async for x in mq]
    assert seq == [0, 1]

    await t


@pytest.mark.asyncio
async def test_overlap() -> None:
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = aiter(mq)
    x1 = await anext(ag1)
    assert x1 == 0

    ag2 = aiter(mq)
    x1, x2 = await asyncio.gather(anext(ag1), anext(ag2))
    assert x1 == 1
    assert x2 == 0

    x1, x2 = await asyncio.gather(anext(ag1, None), anext(ag2))
    assert x1 is None
    assert x2 == 1

    with pytest.raises(StopAsyncIteration):
        await anext(ag2)

    await t


@pytest.mark.asyncio
@pytest.mark.parametrize('use_ctx', [False, True])
async def test_broken_sub_anext(use_ctx: bool) -> None:
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = mq.subscribe()
    if use_ctx:
        with ag1:
            x1 = await anext(ag1)
            assert x1 == 0
    else:
        x1 = await anext(ag1)
        assert x1 == 0
        del ag1

    await asyncio.sleep(0.001)  # Awake worker

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        with pytest.raises(QueueShutdownError):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    with pytest.raises(CancelledError):
        await t


@pytest.mark.asyncio
@pytest.mark.parametrize('use_aclose', [False, True])
async def test_broken_aiter_anext(use_aclose: bool) -> None:
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = aiter(mq)
    x1 = await anext(ag1)
    assert x1 == 0

    if use_aclose:
        await ag1.aclose()
    else:
        del ag1
        await asyncio.sleep(0.001)  # Awake worker

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        with pytest.raises(QueueShutdownError):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    with pytest.raises(CancelledError):
        await t
