__all__ = ['clone_exc', 'hide_frame']

from types import TracebackType


class _HideFrame:
    """Context manager to hide current frame in traceback"""

    def __enter__(self):
        return self

    def __exit__(
        self, tp, val: BaseException | None, tb: TracebackType | None
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


hide_frame = _HideFrame()
