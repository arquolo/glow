from __future__ import annotations

__all__ = ['Reusable']

import asyncio
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from threading import Thread
from typing import Any, Generic, TypeVar

from .concurrency import call_once

_T = TypeVar('_T')
_Make = Callable[[], _T]
_Callback = Callable[[_T], Any]


@call_once
def make_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()
    return loop


async def _await(fn: _Make[_T]) -> _T:
    return fn()


def _trampoline(callback: _Callback[_T], ref: weakref.ref[_T]) -> None:
    if (obj := ref()) is not None:
        callback(obj)


@dataclass
class Reusable(Generic[_T]):
    make: _Make[_T]
    delay: float
    finalize: _Callback[_T] | None = None

    _loop: asyncio.AbstractEventLoop = field(default_factory=make_loop)
    _lock: asyncio.Lock = field(init=False)
    _deleter: asyncio.TimerHandle | None = None
    _box: list[_T] = field(default_factory=list)

    def __post_init__(self):
        coro = _await(asyncio.Lock)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._lock = fut.result()

    def get(self) -> _T:
        """Returns inner object, or recreates it"""
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
