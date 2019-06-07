__all__ = 'sizeof', 'Timed'

import asyncio
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from inspect import isgetsetdescriptor, ismemberdescriptor
from threading import Thread, RLock


def sizeof(obj, seen=None):
    """
    Computes size of object, no matter how complex it is

    Inspired by
    [PySize](https://github.com/bosswissam/pysize/blob/master/pysize.py)
    """
    if seen is None:
        seen = set()
    id_ = id(obj)
    if id_ in seen:
        return 0

    seen.add(id_)
    size = sys.getsizeof(obj)

    if ('numpy' in sys.modules
            and isinstance(obj, sys.modules['numpy'].ndarray)):
        return max(size, obj.nbytes)

    if ('torch' in sys.modules
            and sys.modules['torch'].is_tensor(obj)):
        if not obj.is_cuda:
            size += obj.numel() * obj.element_size()
        return size  # TODO: test, maybe useless when grads are attached

    if isinstance(obj, (str, bytes, bytearray)):
        return size

    # protection from self-referencing
    if hasattr(obj, '__dict__'):
        for d in (vars(cl)['__dict__']
                  for cl in type(obj).mro() if '__dict__' in vars(cl)):
            if isgetsetdescriptor(d) or ismemberdescriptor(d):
                size += sizeof(vars(obj), seen=seen)
            break

    if isinstance(obj, dict):
        size += sum(sizeof(k, seen) + sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, Collection):
        size += sum(sizeof(item, seen=seen) for item in obj)

    if hasattr(obj, '__slots__'):
        size += sum(sizeof(getattr(obj, slot, None), seen=seen)
                    for cl in type(obj).mro()
                    for slot in getattr(cl, '__slots__', ()))
    return size


@dataclass
class Timed:
    _lock = RLock()
    _loop = None
    _marker = object()

    data: object = _marker
    factory: callable = None
    timeout: float = 0
    _atime: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        if self.data is self._marker:
            self.data = self.factory()
        assert self.data is not self._marker

        def start_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        with self._lock:
            if self._loop is not None:
                return
            type(self)._loop = loop = asyncio.new_event_loop()
            Thread(target=start_loop, daemon=True).start()

        async def finalizer():
            while True:
                now = time.time()
                with self._lock:
                    deadline = self._atime + self.timeout
                    if now >= deadline:
                        self.data = self._marker
                        return
                await asyncio.sleep(deadline - now)

        asyncio.run_coroutine_threadsafe(finalizer(), loop=self._loop)

    def get(self):
        with self._lock:
            self._atime = time.time()
            if self.data is self._marker:
                if self.factory is None:
                    raise TimeoutError('Data was already finalized')
                self.data = self.factory()
            return self.data
