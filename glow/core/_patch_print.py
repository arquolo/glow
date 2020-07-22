"""
Patches builtin `print` function to be compatible with `tqdm`.
Adds some thread safety.
"""
__all__ = ['apply']

import builtins
import functools
from threading import RLock
from typing import Any

import wrapt

_print = builtins.print
_lock = RLock()


@functools.wraps(_print)
def locked_print(*args, **kwargs):
    with _lock:
        _print(*args, **kwargs)


def patch_print(tqdm: Any) -> None:
    def new_print(*args, sep=' ', end='\n', file=None, **kwargs) -> None:
        tqdm.tqdm.write(sep.join(map(str, args)), end=end, file=file)

    builtins.print = functools.update_wrapper(new_print, _print)


def apply() -> None:
    builtins.print = locked_print
    wrapt.register_post_import_hook(patch_print, 'tqdm')
