__all__ = ('Timed', )

import asyncio
from asyncio import AbstractEventLoop
from dataclasses import dataclass, field
from threading import Thread

from . import call_once


@call_once
def make_loop():
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    Thread(target=start_loop, args=(loop, ), daemon=True).start()
    return loop


@dataclass
class Timed:
    _lock = asyncio.Lock()
    _loop = None
    _marker = object()

    data: object = _marker
    factory: callable = None
    timeout: float = 0
    _loop: AbstractEventLoop = field(default_factory=make_loop, init=False)
    _finalizer: asyncio.TimerHandle = None

    def __post_init__(self):
        if self.data is self._marker:
            self.data = self.factory()
        assert self.data is not self._marker

        self._finalizer = self._loop.call_later(self.timeout, self._clear)

    def get(self):
        fut = asyncio.run_coroutine_threadsafe(self._get(), loop=self._loop)
        return fut.result()

    def _clear(self):
        self.data = self._marker

    async def _get(self):
        async with self._lock:
            self._finalizer.cancel()
            self._finalizer = self._loop.call_later(self.timeout, self._clear)

            if self.data is self._marker:
                if self.factory is None:
                    raise TimeoutError('Data was already finalized')
                self.data = self.factory()
            return self.data
