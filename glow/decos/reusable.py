__all__ = ('Reusable', )

import asyncio
import weakref
from asyncio import AbstractEventLoop, Lock
from threading import Thread


from . import call_once


@call_once
def make_loop():
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    lock = Lock(loop=loop)
    Thread(target=start_loop, args=(loop, ), daemon=True).start()
    return loop, lock


class Reusable:
    _loop: AbstractEventLoop = None
    _lock: Lock = None

    def __init__(self, factory, finalize=None, timeout: float = 0):
        self.factory = factory
        self.finalize = finalize or (lambda _: None)
        self.timeout = timeout

        self._loop, self._lock = make_loop()
        self._deleter = None

    def get(self):
        """Returns inner object, or recreates it"""
        fut = asyncio.run_coroutine_threadsafe(self._get(), loop=self._loop)
        return fut.result()

    def _delete(self):
        del self.ref

    def _finalize(self, ref):
        obj = ref()
        if obj is not None:
            self.finalize(obj)

    async def _get(self):
        async with self._lock:
            # reschedule finalizer
            if self._deleter is not None:
                self._deleter.cancel()
            self._deleter = self._loop.call_later(self.timeout, self._delete)

            # recreate object, if nessessary
            if not hasattr(self, 'ref'):
                self.ref = self.factory()
                ref = weakref.ref(self.ref)
                weakref.finalize(self.ref, self._finalize, ref)

            return self.ref
