"""
Fix for strange bug in SciPy on Anaconda for Windows
https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats

Combines both:
.. [https://stackoverflow.com/a/39021051]
.. [https://stackoverflow.com/a/44822794]
"""
__all__ = ['apply']

import ctypes
import os
import sys
from contextlib import ExitStack
from pathlib import Path

_FORTRAN_FLAG = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'


def patch_handler() -> None:
    root = Path(sys.prefix)
    if not (root / 'conda-meta').exists():
        return

    # Preload DLLs
    for dllname in ('libmmd.dll', 'libifcoremd.dll'):
        dllpath = root / 'Library' / 'bin' / dllname
        if dllpath.exists():
            ctypes.CDLL(dllpath.as_posix())

    # Picked from (stackoverflow)[https://stackoverflow.com/a/39021051/9868257]
    handler = ctypes.cast(ctypes.windll.kernel32.SetConsoleCtrlHandler,
                          ctypes.c_void_p)

    code: bytearray = ctypes.windll.kernel32.VirtualProtect(  # type: ignore
        handler,
        ctypes.c_size_t(1),
        0x40,
        ctypes.byref(ctypes.c_uint32(0)),
    ) and (ctypes.c_char * 3).from_address(handler.value)  # type: ignore

    patch = b'\xC2\x08\x00' if ctypes.sizeof(ctypes.c_void_p) == 4 else b'\xC3'
    with ExitStack() as stack:
        if code:
            old_code = code[0:len(patch)]
            code[0:len(patch)] = patch
            stack.callback(code.__setitem__, slice(0, len(patch)), old_code)
        import scipy.stats  # noqa: F401


def apply() -> None:
    if sys.platform != 'win32' or _FORTRAN_FLAG in os.environ:
        return
    # Add flag to environment, child processes will inherit it
    os.environ[_FORTRAN_FLAG] = '1'
    try:
        patch_handler()
    except BaseException:  # noqa: PIE786
        pass
