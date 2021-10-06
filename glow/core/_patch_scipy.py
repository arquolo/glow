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

    ptr = ctypes.c_void_p()
    ok = ctypes.windll.kernel32.VirtualProtect(
        ptr, ctypes.c_size_t(1), 0x40, ctypes.byref(ctypes.c_uint32(0)))
    if not ok or (addr := ptr.value) is None:
        return
    code: bytearray = (ctypes.c_char * 3).from_address(addr)  # type: ignore

    patch = b'\xC2\x08\x00' if ctypes.sizeof(ctypes.c_void_p) == 4 else b'\xC3'
    # Meaningless if scipy.stats is loaded
    if code and 'scipy.stats' not in sys.modules:
        patch_size = len(patch)
        old_code, code[:patch_size] = code[:patch_size], patch
        try:
            import scipy.stats  # noqa: F401
        finally:
            code[:patch_size] = old_code


def apply() -> None:
    if sys.platform != 'win32' or _FORTRAN_FLAG in os.environ:
        return
    # Add flag to environment, child processes will inherit it
    os.environ[_FORTRAN_FLAG] = '1'
    try:
        patch_handler()
    except BaseException:  # noqa: PIE786
        pass
