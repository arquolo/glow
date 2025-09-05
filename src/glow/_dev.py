__all__ = ['declutter_tb']

from types import CodeType


def declutter_tb[E: BaseException](e: E, code: CodeType) -> E:
    tb = e.__traceback__

    # Drop outer to `code` frames
    while tb:
        if tb.tb_frame.f_code is code:  # Has reached target frame
            e.__traceback__ = tb
            return e

        tb = tb.tb_next
    return e
