# flake8: noqa
"""Functional Python tools"""

from importlib import import_module
from typing import TYPE_CHECKING

from . import _patch_len, _patch_print, _patch_scipy
from ._array import aceil, afloor, apack, around
from ._coro import as_actor, coroutine, summary
from ._debug import lock_seed, trace, trace_module, whereami
from ._import_hook import register_post_import_hook, when_imported
from ._more import (as_iter, chunked, eat, groupby, ichunked, ilen, roundrobin,
                    windowed)
from ._parallel import (buffered, get_executor, map_n, map_n_dict,
                        max_cpu_count, starmap_n)
from ._profile import memprof, time_this, timer
from ._repr import countable, mangle, repr_as_obj, si, si_bin
from ._sizeof import sizeof
from ._uuid import Uid
from .wrap import (Reusable, call_once, memoize, shared_call, streaming,
                   threadlocal, weak_memoize)

if TYPE_CHECKING:
    from ._ic import ic, ic_repr
else:
    _exports = {'._ic': ['ic', 'ic_repr']}
    _submodule_by_name = {
        name: modname for modname, names in _exports.items() for name in names
    }

    def __getattr__(name: str):
        if mod := _submodule_by_name.get(name):
            mod = import_module(mod, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__


__all__ = [
    'Reusable', 'Uid', 'aceil', 'afloor', 'around', 'as_actor', 'as_iter',
    'buffered', 'call_once', 'chunked', 'coroutine', 'countable', 'eat',
    'get_executor', 'groupby', 'ic', 'ic_repr', 'ichunked', 'ilen',
    'lock_seed', 'mangle', 'map_n', 'map_n_dict', 'max_cpu_count', 'memoize',
    'memprof', 'apack', 'register_post_import_hook', 'repr_as_obj',
    'roundrobin', 'shared_call', 'si', 'si_bin', 'sizeof', 'starmap_n',
    'streaming', 'summary', 'threadlocal', 'time_this', 'timer', 'trace',
    'trace_module', 'weak_memoize', 'when_imported', 'whereami', 'windowed'
]

_patch_print.apply()
_patch_len.apply()
_patch_scipy.apply()
