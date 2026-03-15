__all__ = ['Reusable']

import asyncio
import weakref
from dataclasses import dataclass, field
from functools import partial
from threading import Thread
from typing import Generic, TypeVar

from ._cache import memoize
from ._types import Callback, Get

_T = TypeVar('_T')


@memoize()
def make_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()
    return loop


async def _await(fn: Get[_T]) -> _T:
    return fn()


def _trampoline(callback: Callback[_T], ref: weakref.ref[_T]) -> None:
    if (obj := ref()) is not None:
        callback(obj)


@dataclass
class Reusable(Generic[_T]):
    make: Get[_T]
    delay: float
    finalize: Callback[_T] | None = None

    _loop: asyncio.AbstractEventLoop = field(default_factory=make_loop)
    _deleter: asyncio.TimerHandle | None = None
    _box: list[_T] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __call__(self) -> _T:
        """Retrieve or recreate inner object."""
        fut = asyncio.run_coroutine_threadsafe(self._get(), self._loop)
        return fut.result()

    async def _get(self) -> _T:
        async with self._lock:
            # reschedule finalizer
            if self._deleter is not None:
                self._deleter.cancel()
            self._deleter = self._loop.call_later(self.delay, self._box.clear)

            # recreate object, if nessessary
            if not self._box:
                obj = await asyncio.to_thread(self.make)
                if self.finalize is not None:
                    weakref.ref(obj, partial(_trampoline, self.finalize))
                self._box.append(obj)

            return self._box[0]
