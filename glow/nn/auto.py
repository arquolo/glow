__all__ = ('Input', 'Model')

import torch
from torch.nn import Module, ModuleDict

from dataclasses import dataclass, field
from typing import List

from ..core import countable


class Model(ModuleDict):
    def __init__(self, inputs: 'List[Input]', outputs: 'List[Input]'):
        super().__init__()

        count = countable()
        self.inputs = [str(count(x)) for x in inputs]

        roots = list(inputs)
        while roots:
            root = roots.pop()
            roots.extend(root.leaves)
            if all(leaf.module is None for leaf in root.leaves):
                continue
            self[str(count(root))] = ModuleDict({
                str(count(leaf)): leaf.module for leaf in root.leaves
            })

        self.outputs = [str(count(x)) for x in outputs]

    def forward(self, *inputs):
        state = dict(zip(self.inputs, inputs))
        while set(self.outputs) - set(state):
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
    module: Module = None
    leaves: 'List[Input]' = field(default_factory=list, init=False)

    def __or__(self, node):
        channels = self.channels

        if isinstance(node, ModuleWrapper):
            if node.args and isinstance(node.args[0], int):
                channels = node.args[0]
            node = node.module(self.channels, *node.args, **node.kwargs)

        elif not isinstance(node, Module):
            raise TypeError(
                f'Expected either ModuleWrapper, or torch.nn.Module class,' +
                f' got {type(node)}')

        leaf = Input(channels, node)
        self.leaves.append(leaf)
        return leaf


@dataclass
class ModuleWrapper:
    module: type
    args: tuple = None
    kwargs: dict = None

    def __call__(self, *_args, **_kwargs):
        self.args = _args
        self.kwargs = _kwargs
        return self


def __dir__():
    return __all__ + tuple(dir(torch.nn))


def __getattr__(name):
    if name not in dir(torch.nn):
        return globals()[name]
    return ModuleWrapper(getattr(torch.nn, name))
