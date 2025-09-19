__all__ = ['hide_frame']


class _HideFrame:
    """Context manager to hide current frame in traceback"""

    def __enter__(self):
        return self

    def __exit__(self, tp, val, tb):
        if tp is None:
            return True
        if tb := val.__traceback__:
            val.__traceback__ = tb.tb_next
        return False


hide_frame = _HideFrame()
