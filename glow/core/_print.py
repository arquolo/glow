"""Patches builtin `print` function to be compatible with `tqdm`"""
__all__ = ()

import builtins
import functools
import threading
from typing import Any

import wrapt

_print = builtins.print
_lock = threading.RLock()


@functools.wraps(_print)
def locked_print(*args, **kwargs):
    with _lock:
        _print(*args, **kwargs)


@wrapt.when_imported('tqdm')
def patch_print(tqdm: Any) -> None:
    @functools.wraps(_print)
    def new_print(*args, sep=' ', end='\n', file=None, **kwargs) -> None:
        tqdm.tqdm.write(sep.join(map(str, args)), end=end, file=file)

    builtins.print = new_print


builtins.print = locked_print
