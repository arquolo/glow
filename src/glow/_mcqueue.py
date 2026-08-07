__all__ = ['MulticastQueue']

from asyncio import CancelledError, Future
from collections.abc import AsyncGenerator, Iterable, MutableSet
from typing import Literal, Self
from weakref import finalize

from ._types import Get


class QueueShutdownError(Exception):
    """Raised on access to terminated Queue."""


class MulticastQueue[T]:
    """Single producer, multiple consumer queue.

    Each consumer gets each message put by producer.
    Late joined consumer gets all messages from the beginning.

    Usage:

        mq = MulticastQueue()
        async def worker():
            with mq:  # or just mq.close() after last mq.put
                for x in range(3):
                    mq.put(x)

        t = asyncio.create_task(worker())
        assert [x async for x in mq] == [0, 1, 2]
        assert [x async for x in mq] == [0, 1, 2]

    """

    def __init__(self) -> None:
        self._buf: list[T] = []
        self._putters = set[Future[None]]()
        self._getters = set[Future[None]]()
        self._state: Literal['running', 'closed', 'terminated'] = 'running'
        self._nsubs = 0

    # ------------------------------- consumer -------------------------------

    async def __aiter__(self) -> AsyncGenerator[T]:
        """Subscribe to queue and iterate over its items."""
        # NOTE: finally doesn't work unless `aclose` is manually called,
        # or anything is `await`ed (even via `asyncio.sleep(0.001)`)
        with self.subscribe() as it:
            async for x in it:
                yield x

    def subscribe(self) -> '_MulticastQueueIterator[T]':
        """Subscribe to queue to get items from its beginning."""
        self._nsubs += 1

        def unsubsribe() -> None:
            self._nsubs -= 1
            if self._state in ('terminated', 'closed'):  # Already stopped
                return
            if self._nsubs:  # Is not the last subscriber
                return
            self._state = 'terminated'
            if self._getters:
                raise RuntimeError(
                    f'{len(self._getters)} non subscribed getters'
                )
            _cancel_all(self._putters, msg='No waiters to store values for')

        return _MulticastQueueIterator(self, unsubsribe)

    async def get(self, idx: int) -> T:
        if idx >= len(self._buf):
            if self._state == 'terminated':  # Closed before finish
                raise QueueShutdownError

            if self._state == 'closed':  # Finished
                raise IndexError

            await _step(
                None,
                wakeup=self._putters,
                wait_for=self._getters,
                cancel_with='Get was interrupted',
            )

        return self._buf[idx]

    # ------------------------------- producer -------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Close queue to mark a successful end of production."""
        if self._state == 'running':
            self._state = 'closed'
            if self._putters:
                raise RuntimeError(
                    'Cannot close queue while there are '
                    f'{len(self._putters)} active writes'
                )
            _cancel_all(self._getters, msg='Queue is finalized')

    async def put(self, value: T) -> None:
        """Put new value to queue."""
        if self._state in ('closed', 'terminated'):
            raise QueueShutdownError

        self._buf.append(value)
        await _step(
            None,
            wakeup=self._getters,
            wait_for=self._putters,
            cancel_with='Put was interrupted',
        )


class _MulticastQueueIterator[T]:
    def __init__(self, mq: MulticastQueue[T], finalizer: Get[None]) -> None:
        self._mq = mq
        self._pos = 0
        self.close = finalize(self, finalizer)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._pos}: ...)'

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            value = await self._mq.get(self._pos)
        except (IndexError, CancelledError) as exc:
            raise StopAsyncIteration from exc
        else:
            self._pos += 1
            return value


async def _step[T](
    value: T,
    /,
    *,
    wakeup: Iterable[Future[T]],
    wait_for: MutableSet[Future[T]],
    cancel_with: str | None = None,
) -> T:
    _wakeup(wakeup, value)
    return await _wait_for(wait_for, cancel_with=cancel_with)


def _wakeup[T](fs: Iterable[Future[T]], value: T) -> None:
    for f in fs:
        if not f.done():
            f.set_result(value)  # Release blocked


async def _wait_for[T](
    fs: MutableSet[Future[T]], cancel_with: str | None = None
) -> T:
    f = Future[T]()
    fs.add(f)
    try:
        return await f  # Acquire to block
    finally:
        f.cancel(cancel_with)
        fs.discard(f)


def _cancel_all(waiters: Iterable[Future], msg: str | None = None) -> None:
    for f in waiters:
        f.cancel(msg)
