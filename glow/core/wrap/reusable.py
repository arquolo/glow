__all__ = ('Reusable', )

import asyncio
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Generic, List, Optional, TypeVar
from typing_extensions import Protocol

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)


def make_loop() -> asyncio.AbstractEventLoop:
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    threading.Thread(target=start_loop, args=(loop, ), daemon=True).start()
    return loop


def _finalize(_: object) -> None:
    pass


class _Factory(Protocol[_T_co]):
    def __call__(self) -> _T_co:
        ...


@dataclass
class Reusable(Generic[_T]):
    factory: _Factory[_T]
    finalize: Callable[[_T], None] = _finalize
    delay: float = 0.0

    _loop: ClassVar[asyncio.AbstractEventLoop] = make_loop()
    _lock: asyncio.Lock = field(
        default_factory=lambda: asyncio.Lock(loop=Reusable._loop))
    _deleter: Optional[asyncio.TimerHandle] = None
    _box: List[_T] = field(default_factory=list)

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
            self._deleter = self._loop.call_later(self.delay, self._box.clear)

            # recreate object, if nessessary
            if not self._box:
                obj = self.factory()
                weakref.finalize(obj, self._finalize, weakref.ref(obj))
                self._box.append(obj)

            return self._box[0]
