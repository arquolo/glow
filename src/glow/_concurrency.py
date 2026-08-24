__all__ = ['threadlocal']

import threading
from collections.abc import Callable
from functools import update_wrapper

from ._types import Get


def threadlocal[**P, R](
    fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> Get[R]:
    """Create thread-local singleton factory function (functools.partial)."""
    local_ = threading.local()

    def wrapper() -> R:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)
