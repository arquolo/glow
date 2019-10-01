__all__ = ('param_count', 'plot_model')

from contextlib import ExitStack
from functools import partial
from typing import Dict

import graphviz
import torch
from torch.autograd import Function

from ..core import decimate, mangle


def id_(x):
    return hex(id(x))


def as_tuple(xs):
    if xs is None:
        return ()
    if isinstance(xs, tuple):
        return tuple(x for x in xs if x is not None)
    return (xs, )


class Builder:
    flat = False

    def __init__(self, inputs: set, params: dict):
        self.inputs = inputs
        self.params = params

        self._mangle = mangle()
        self._seen: Dict[Function, str] = {}
        self._shapes: Dict[Function, str] = {}
        self.stack = [graphviz.Digraph(
            name='root',
            graph_attr={
                'rankdir': 'LR',
                'newrank': 'true',
                'color': 'lightgrey',
            },
            edge_attr={
                'labelfloat': 'true',
            },
            node_attr={
                'shape': 'box',
                'style': 'filled',
                'fillcolor': 'lightgrey',
                'fontsize': '12',
                'height': '0.2',
                'ranksep': '0.1',
            },
        )]

    def _sized(self, var):
        if sum(1 for s in var.shape if s != 1) <= 1:
            return f'{tuple(var.shape)}'
        return '{}\n{:.0f}{}'.format(tuple(var.shape), *decimate(var.numel()))

    def _add_node(self, grad):
        root = self.stack[-1]
        if not hasattr(grad, 'variable'):
            label = type(grad).__name__.replace('Backward', '')
            if grad in self._shapes:
                label += f'\n=> {tuple(self._shapes[grad])}'
            root.node(id_(grad), label)
            return False

        var, label = grad.variable, ''
        try:
            name = self.params[id(var)] + '\n'
            label = (name.partition if self.flat else name.rpartition)('.')[-1]
        except KeyError:
            root = self.stack[0]  # unnamed, that's why external

        color = 'yellow' if id(var) in self.inputs else 'lightblue'
        root.node(id_(grad), label + self._sized(var), fillcolor=color)
        return True

    def _traverse(self, grad, depth=0):
        if grad is None or grad in self._seen:
            return

        root = self.stack[-1]
        self._seen[grad] = root.name
        if self._add_node(grad):
            yield (depth - 1, None, grad)
            return

        for ch, _ in getattr(grad, 'next_functions', ()):
            if ch is None:
                continue
            yield from self._traverse(ch, depth + 1)

            tail, head = self._seen.get(ch), root.name
            if tail is not None and not (head.startswith(tail) or
                                         tail.startswith(head)):
                yield (depth, ch, grad)  # leafs, yield for depth-check
                continue

            name = self.params.get(id(getattr(ch, 'variable', None)))
            if not self.flat and name and name.rpartition('.')[0] == head:
                with self.stack[-1].subgraph() as s:
                    s.attr(rank='same')
                    s.edge(id_(ch), id_(grad))  # same module, same rank
            else:
                self.stack[0].edge(id_(ch), id_(grad))

        for var in getattr(grad, 'saved_tensors', ()) or ():
            if torch.is_tensor(var):
                root.node(id_(var), self._sized(var), fillcolor='orange')
                self.stack[0].edge(id_(var), id_(grad))

    def _mark(self, ts):
        edges = []
        for t in as_tuple(ts):
            if t.grad_fn is not None:
                self._shapes[t.grad_fn] = t.shape
                edges.extend(self._traverse(t.grad_fn))
        if not edges:
            return

        max_depth = max(depth for depth, *_ in edges) + 1
        for depth, tail, head in edges:  # inter-module edges
            if tail is not None:
                minlen = None if self.flat else f'{max_depth - depth}'
                self.stack[0].edge(id_(tail), id_(head), minlen=minlen)

    def forward_pre(self, name, module, xs):
        self._mark(xs)
        # -------- start node --------
        if self.flat:
            return
        scope = graphviz.Digraph(name=self._mangle(name))
        scope.attr(label=f'{name.split(".")[-1]}:{type(module).__name__}')
        self.stack.append(scope)

    def forward(self, module, _, ys):
        self._mark(ys)
        if self.flat:
            return
        cluster = self.stack.pop(-1)
        cluster.name = f'cluster_{cluster.name}'
        self.stack[-1].subgraph(cluster)
        # -------- end node --------


def plot_model(model: torch.nn.Module, *input_shapes: tuple, device='cpu'):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    inputs = (torch.zeros(1, *shape, device=device) for shape in input_shapes)
    inputs = [inp.requires_grad_() for inp in inputs]
    params = model.state_dict(prefix='root.', keep_vars=True)
    hk = Builder(
        {id(var) for var in inputs},
        {id(var): name for name, var in params.items()},
    )
    with ExitStack() as stack:
        model.dump_patches = True
        for name, m in model.named_modules(prefix='root'):
            for handle in (
                m.register_forward_pre_hook(partial(hk.forward_pre, name)),
                m.register_forward_hook(hk.forward),
            ):
                stack.callback(handle.remove)
        model(*inputs)

    dot = hk.stack.pop()
    assert not hk.stack

    dot.filename = getattr(model, 'name', type(model).__name__)
    dot.directory = 'graphs'
    dot.format = 'svg'

    size_min = 12
    scale_factor = .15
    size = max(size_min, len(dot.body) * scale_factor)

    dot.graph_attr.update(size=f'{size},{size}')
    dot.render(cleanup=True)
    return dot


def param_count(module: torch.nn.Module):
    return '{:.0f}{}'.format(
        *decimate(sum(p.numel() for p in module.parameters()), base=1000)
    )