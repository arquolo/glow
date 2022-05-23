from importlib import import_module
from typing import TYPE_CHECKING

from ._loader import make_loader
from ._stepper import Stepper
from .amp import get_amp_context
from .driver import get_gpu_state
from .modules import (Dropsample, Ensemble, Noise, SplitAttention,
                      SqueezeExcitation, Upscale2d)
from .optimizers import SGDW, AdamW, RAdam
from .util import device, dump_to_onnx, frozen, inference, param_count, profile

__all__ = [
    'AdamW', 'Dropsample', 'Ensemble', 'Noise', 'RAdam', 'SGDW',
    'SplitAttention', 'SqueezeExcitation', 'Stepper', 'Upscale2d', 'device',
    'dump_to_onnx', 'frozen', 'get_amp_context', 'get_gpu_state', 'inference',
    'make_loader', 'param_count', 'plot_model', 'profile'
]

_exports = {
    '.plot': ['plot_model'],
}
_submodule_by_name = {
    name: modname for modname, names in _exports.items() for name in names
}

if TYPE_CHECKING:
    from .plot import plot_model
else:

    def __getattr__(name: str):
        if modname := _submodule_by_name.get(name):
            mod = import_module(modname, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__
