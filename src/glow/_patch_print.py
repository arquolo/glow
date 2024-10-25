"""
Patches builtin `print` function to be compatible with `tqdm`.
Adds some thread safety.
"""

__all__ = ['apply']

import builtins
from functools import update_wrapper, wraps
from threading import RLock
from typing import Protocol

from ._import_hook import register_post_import_hook

_print = builtins.print
_lock = RLock()


@wraps(_print)
def locked_print(*args, **kwargs) -> None:
    with _lock:
        _print(*args, **kwargs)


def patch_print(module) -> None:
    # Create blank to force initialization of cls._lock and cls._instances
    tqdm = module.tqdm
    tqdm(disable=True)

    def tqdm_print(
        *values,
        sep: str | None = ' ',
        end: str | None = '\n',
        file: _SupportsWrite | None = None,
        flush: bool = False,
    ) -> None:
        if sep is None:
            sep = ' '
        if end is None:
            end = '\n'
        tqdm.write(sep.join(map(str, values)), end=end, file=file)

    builtins.print = update_wrapper(tqdm_print, _print)  # type: ignore


def apply() -> None:
    builtins.print = locked_print  # type: ignore
    register_post_import_hook(patch_print, 'tqdm')


class _SupportsWrite(Protocol):
    def write(self, s: str, /) -> object: ...
