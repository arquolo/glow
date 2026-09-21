__all__ = [
    'clone_exc',
    'declutter_tb',
    'hide_frame',
    'lock_seed',
]

import copy
import os
import random
from types import CodeType, FrameType
from typing import Self

import numpy as np

from ._import_hook import register_post_import_hook


class _HideFrame:
    """Context manager to hide current frame in traceback"""

    def __init__(self, nframes: int = 1) -> None:
        self._nframes = nframes

    def __call__(self, nframes: int) -> Self:
        return self.__class__(nframes)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, tp, val: BaseException | None, tb) -> None:
        if val is not None:
            drop_tb_frames(val, self._nframes)


def drop_tb_frames(exc: BaseException, n: int) -> None:
    for _ in range(n):
        if not exc.__traceback__:
            return
        exc.__traceback__ = exc.__traceback__.tb_next  # Drop outer frame


def clone_exc[E: BaseException](exc: E) -> E:
    return copy.copy(exc)


def declutter_tb(e: BaseException, code: CodeType) -> None:
    tb = e.__traceback__

    # Drop frames until `code` frame is reached
    while tb:
        if tb.tb_frame.f_code is code:
            e.__traceback__ = tb
            return
        tb = tb.tb_next


hide_frame = _HideFrame()


# ---------------------------------------------------------------------------


def frame_key(frame: FrameType | None) -> tuple[str, int] | None:
    return (frame.f_code.co_filename, frame.f_lineno) if frame else None


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
