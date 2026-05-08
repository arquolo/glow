__all__ = [
    'clone_exc',
    'declutter_tb',
    'hide_frame',
    'lock_seed',
    'whereami',
]

import gc
import os
import random
from inspect import currentframe, getmodule, isfunction
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np

from ._cache import memoize
from ._import_hook import register_post_import_hook

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import CodeType, FrameType, TracebackType


def _frame_hash(frame: 'FrameType') -> tuple[str, int]:
    return frame.f_code.co_filename, frame.f_lineno


@memoize(100, policy='lru', key_fn=_frame_hash)
def _get_source(frame: 'FrameType') -> str:
    # Get source module name
    modname = (
        spec.name
        if (module := getmodule(frame)) and (spec := module.__spec__)
        else '__main__'
    )

    # Get source code name (method or function name)
    code = frame.f_code
    codename = next(
        (f.__qualname__ for f in gc.get_referrers(code) if isfunction(f)),
        code.co_name,
    )
    if codename == '<module>':
        codename = ''

    return f'{modname}:{codename}:{frame.f_lineno}'


def _get_source_calls(frame: 'FrameType | None') -> 'Iterator[str]':
    while frame:
        yield _get_source(frame)
        if frame.f_code.co_name == '<module>':  # Stop on module-level scope
            return
        frame = frame.f_back


def stack(skip: int = 0, limit: int | None = None) -> 'Iterator[str]':
    """Return iterator of FrameInfos, stopping on module-level scope."""
    frame = currentframe()
    calls = _get_source_calls(frame)
    calls = islice(calls, skip + 1, None)  # Skip 'skip' outerless frames
    if not limit:
        return calls
    if limit < 0:
        return iter(list(calls)[:limit])
    return islice(calls, limit)  # Keep at most `limit` outer frames


def whereami(skip: int = 0, limit: int | None = None) -> str:
    calls = stack(skip + 1, limit)
    return ' -> '.join(reversed([*calls]))


class _HideFrame:
    """Context manager to hide current frame in traceback"""

    def __enter__(self):
        return self

    def __exit__(
        self, tp, val: BaseException | None, tb: 'TracebackType | None'
    ):
        if val is not None:
            tb = val.__traceback__ or tb
            if tb:
                val.__traceback__ = tb.tb_next  # Drop outer traceback frame


def clone_exc[E: BaseException](exc: E) -> E:
    new_exc = type(exc)(*exc.args)
    new_exc.__cause__ = exc.__cause__
    new_exc.__context__ = exc.__context__
    new_exc.__traceback__ = exc.__traceback__
    return new_exc


def declutter_tb(e: BaseException, code: 'CodeType') -> None:
    tb = e.__traceback__

    # Drop frames until `code` frame is reached
    while tb:
        if tb.tb_frame.f_code is code:
            e.__traceback__ = tb
            return
        tb = tb.tb_next


hide_frame = _HideFrame()


# ---------------------------------------------------------------------------


def lock_seed(seed: int) -> None:
    """Set seed for all modules: random/numpy/torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    def _torch_seed(torch) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    register_post_import_hook(_torch_seed, 'torch')
