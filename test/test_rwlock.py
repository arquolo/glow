import asyncio

import pytest

from glow import RwLock


@pytest.mark.asyncio
async def test_multiple_readers() -> None:
    lock = RwLock()
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def read() -> None:
        nonlocal entered
        async with lock.read():
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()

    tasks = [asyncio.create_task(read()) for _ in range(2)]
    await asyncio.wait_for(both_entered.wait(), 1)
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_writes_are_exclusive() -> None:
    lock = RwLock()
    active = 0
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def write(first: bool) -> None:
        nonlocal active
        async with lock.write():
            active += 1
            assert active == 1
            if first:
                first_entered.set()
                await release_first.wait()
            active -= 1

    first = asyncio.create_task(write(True))
    await asyncio.wait_for(first_entered.wait(), 1)
    second = asyncio.create_task(write(False))
    await asyncio.sleep(0)
    assert not second.done()

    release_first.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_waiting_writer_blocks_new_readers() -> None:
    lock = RwLock()
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    second_reader_entered = asyncio.Event()

    async def first_reader() -> None:
        async with lock.read():
            first_reader_entered.set()
            await release_first_reader.wait()

    async def writer() -> None:
        async with lock.write():
            writer_entered.set()
            await release_writer.wait()

    async def second_reader() -> None:
        async with lock.read():
            second_reader_entered.set()

    reader1 = asyncio.create_task(first_reader())
    await asyncio.wait_for(first_reader_entered.wait(), 1)
    write_task = asyncio.create_task(writer())
    await asyncio.sleep(0)
    reader2 = asyncio.create_task(second_reader())
    await asyncio.sleep(0)

    assert not writer_entered.is_set()
    assert not second_reader_entered.is_set()

    release_first_reader.set()
    await asyncio.wait_for(writer_entered.wait(), 1)
    assert not second_reader_entered.is_set()

    release_writer.set()
    await asyncio.wait_for(second_reader_entered.wait(), 1)
    await asyncio.gather(reader1, write_task, reader2)


@pytest.mark.asyncio
async def test_cancelled_writer_unblocks_readers() -> None:
    lock = RwLock()
    release_first_reader = asyncio.Event()
    first_reader_entered = asyncio.Event()
    second_reader_entered = asyncio.Event()

    async def first_reader() -> None:
        async with lock.read():
            first_reader_entered.set()
            await release_first_reader.wait()

    async def writer() -> None:
        async with lock.write():
            pytest.fail('cancelled writer entered the lock')

    async def second_reader() -> None:
        async with lock.read():
            second_reader_entered.set()

    reader1 = asyncio.create_task(first_reader())
    await asyncio.wait_for(first_reader_entered.wait(), 1)
    write_task = asyncio.create_task(writer())
    await asyncio.sleep(0)
    reader2 = asyncio.create_task(second_reader())
    await asyncio.sleep(0)
    assert not second_reader_entered.is_set()

    write_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await write_task
    await asyncio.wait_for(second_reader_entered.wait(), 1)

    release_first_reader.set()
    await asyncio.gather(reader1, reader2)
