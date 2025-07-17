__all__ = ['Reusable']

import asyncio
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from threading import Thread

from ._cache import memoize

type _Get[T] = Callable[[], T]
type _Callback[T] = Callable[[T], object]


@memoize()
def make_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()
    return loop


async def _await[T](fn: _Get[T]) -> T:
    return fn()


def _trampoline[T](callback: _Callback[T], ref: weakref.ref[T]) -> None:
    if (obj := ref()) is not None:
        callback(obj)


@dataclass
class Reusable[T]:
    make: _Get[T]
    delay: float
    finalize: _Callback[T] | None = None

    _loop: asyncio.AbstractEventLoop = field(default_factory=make_loop)
    _deleter: asyncio.TimerHandle | None = None
    _box: list[T] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __call__(self) -> T:
        """Returns inner object, or recreates it"""
        fut = asyncio.run_coroutine_threadsafe(self._get(), self._loop)
        return fut.result()

    async def _get(self) -> T:
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
