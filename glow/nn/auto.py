__all__ = ('Input', 'Model')

import functools
from contextlib import suppress
from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch
from torch import nn

from ..api import get_wild_imports
from ..core import countable


class Model(nn.ModuleDict):
    def __init__(self, inputs: List['Input'], outputs: List['Input']):
        super().__init__()

        count = countable()
        self.inputs = [f'{count(x)}' for x in inputs]

        roots = list(inputs)
        while roots:
            root = roots.pop()
            roots.extend(root.leaves)
            if all(leaf.module is None for leaf in root.leaves):
                continue
            self[f'{count(root)}'] = nn.ModuleDict({
                str(count(leaf)): leaf.module for leaf in root.leaves
            })

        self.outputs = [str(count(x)) for x in outputs]

    def forward(self, *inputs):
        state = dict(zip(self.inputs, inputs))
        while {*self.outputs} - {*state}:
            for root_id in list(state):
                if root_id in self:
                    tensor = state.pop(root_id)
                    state.update({
                        leaf_id: module(tensor)
                        for leaf_id, module in self[root_id].items()
                    })

        return tuple(state[o] for o in self.outputs)


@dataclass
class Input:
    channels: int = 0
    module: Optional[nn.Module] = None
    leaves: List['Input'] = field(default_factory=list, init=False)

    def __or__(self, node):
        channels = self.channels

        if isinstance(node, ModuleWrapper):
            if node.args and isinstance(node.args[0], int):
                channels = node.args[0]
            node = node.module_type(self.channels, *node.args, **node.kwargs)

        elif not isinstance(node, nn.Module):
            raise TypeError(
                f'Expected either ModuleWrapper, or torch.nn.Module class,' +
                f' got {type(node)}')

        leaf = Input(channels, node)
        self.leaves.append(leaf)
        return leaf


@dataclass
class ModuleWrapper:
    module_type: Type[nn.Module]
    args: Optional[tuple] = None
    kwargs: Optional[dict] = None

    def __call__(self, *_args, **_kwargs):
        self.args = _args
        self.kwargs = _kwargs
        return self


def __dir__():
    return __all__ + get_wild_imports(nn)


def __getattr__(name):
    if name in get_wild_imports(nn):
        return ModuleWrapper(getattr(nn, name))

    with suppress(KeyError):
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
