__all__ = ('Input', 'Model')

import torch

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List


class Model(torch.nn.ModuleDict):
    def __init__(self, inputs: 'List[Input]', outputs: 'List[Input]'):
        super().__init__()
        self.tree = defaultdict(list)

        self.inputs = [id(x) for x in inputs]
        self.outputs = [id(x) for x in outputs]

        roots = inputs
        while roots:
            root = roots.pop()
            for child in root.children:
                module = child.module
                self.tree[id(root)].append((id(child), id(module)))
                self.add_module(f'node_{id(module):x}', module)
                roots.append(child)

    def forward(self, *inputs):
        roots = self.inputs.copy()
        state = {inp: x for inp, x in zip(self.inputs, inputs)}
        while roots:
            root_id = roots.pop()
            for child_id, module_id in self.tree[root_id]:
                state[child_id] = self[f'node_{module_id:x}'](state[root_id])
                roots.append(child_id)

        return tuple(state[o] for o in self.outputs)


@dataclass
class Input:
    channels: int = 0
    module: torch.nn.Module = None
    children: 'List[Input]' = field(default_factory=list, init=False)

    def __or__(self, node):
        channels = self.channels

        if isinstance(node, ModuleWrapper):
            if node.args and isinstance(node.args[0], int):
                channels = node.args[0]
            node = node.module(self.channels, *node.args, **node.kwargs)

        elif not isinstance(node, torch.nn.Module):
            raise TypeError(f'Expected ModuleWrapper or torch.nn.Module class,'
                            f' got {type(node)}')

        child = Input(channels, node)
        self.children.append(child)
        return child


@dataclass
class ModuleWrapper:
    module: type
    args: tuple = None
    kwargs: dict = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


def __dir__():
    return __all__ + tuple(dir(torch.nn))


def __getattr__(name):
    if name not in dir(torch.nn):
        return globals()[name]
    return ModuleWrapper(getattr(torch.nn, name))
