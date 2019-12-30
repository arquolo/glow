__all__ = ('Reusable', )

import asyncio
import threading
import weakref
from typing import Any, Callable, Generic, List, Optional, TypeVar

from .thread import call_once

_T = TypeVar('_T')


@call_once
def make_loop() -> asyncio.AbstractEventLoop:
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    threading.Thread(target=start_loop, args=(loop, ), daemon=True).start()
    return loop


def _finalize(_: Any) -> None:
    pass


class Reusable(Generic[_T]):
    def __init__(self,
                 factory: Callable[[], _T],
                 finalize: Callable[[_T], None] = _finalize,
                 timeout=0.0):
        self.factory = factory
        self.finalize = finalize
        self.timeout = timeout

        self._loop = make_loop()
        self._lock = asyncio.Lock(loop=self._loop)
        self._deleter: Optional[asyncio.TimerHandle] = None
        self._bx: List[_T] = []

    def get(self) -> _T:
        """Returns inner object, or recreates it"""
        fut = asyncio.run_coroutine_threadsafe(self._get(), loop=self._loop)
        return fut.result()

    def _finalize(self, ref):
        obj: Optional[_T] = ref()
        if obj is not None:
            self.finalize(obj)

    async def _get(self) -> _T:
        async with self._lock:
            # reschedule finalizer
            if self._deleter is not None:
                self._deleter.cancel()
            self._deleter = self._loop.call_later(self.timeout, self._bx.clear)

            # recreate object, if nessessary
            if not self._bx:
                obj = self.factory()
                weakref.finalize(obj, self._finalize, weakref.ref(obj))
                self._bx.append(obj)

            return self._bx[0]
