__all__ = [
    'clone_exc',
    'declutter_tb',
    'hide_frame',
    'lock_seed',
]

import copy
import os
import random
from typing import TYPE_CHECKING, Self

import numpy as np

from ._import_hook import register_post_import_hook

if TYPE_CHECKING:
    from types import CodeType, TracebackType


class _HideFrame:
    """Context manager to hide current frame in traceback"""

    def __init__(self, nframes: int = 1) -> None:
        self._nframes = nframes

    def __call__(self, nframes: int) -> Self:
        return self.__class__(nframes)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        tp: type[BaseException] | None,
        val: BaseException | None,
        tb: 'TracebackType | None',
    ) -> None:
        if val is None:
            return
        tb = val.__traceback__ or tb
        for _ in range(self._nframes):
            if not tb:
                break
            tb = tb.tb_next  # Drop outer traceback frame
        val.__traceback__ = tb


def clone_exc[E: BaseException](exc: E) -> E:
    return copy.copy(exc)


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
