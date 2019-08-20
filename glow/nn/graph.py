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

    def __init__(self, inputs, params):
        self.inputs = inputs
        self.params = params

        self._mangle = mangle()
        self._seen: Dict[Function, str] = {}
        self._shapes: Dict[Function, str] = {}
        self._stack = [graphviz.Digraph(
            graph_attr={
                'rankdir': 'LR',
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

    @property
    def dot(self):
        return self._stack[0]

    def _process_var(self, grad):
        root, var = self._stack[-1], grad.variable
        try:
            label = self.params[id(var)].split('.')[-1] + '\n'
        except KeyError:
            label, root = '', self.dot

        label += f'\n{tuple(var.shape)}'
        label += '{:.2g}{}'.format(*decimate(var.numel() * var.element_size()))
        color = 'yellow' if id(var) in map(id, self.inputs) else 'lightblue'
        root.node(id_(grad), label.strip, fillcolor=color)

    def _traverse(self, grad, depth=0):
        if grad is None or grad in self._seen:
            return

        root = self._stack[-1]
        self._seen[grad] = root.name

        if hasattr(grad, 'variable'):
            yield (depth - 1, None, grad)
            self._process_var(grad)
            return

        label = type(grad).__name__.replace('Backward', '')
        shape = self._shapes.get(grad)
        if shape is not None:
            label += f'\n=> {tuple(shape)}'
        root.node(id_(grad), label)

        for ch, _ in getattr(grad, 'next_functions', ()):
            if ch is None:
                continue
            yield from self._traverse(ch, depth=depth + 1)

            tail = self._seen.get(ch)
            if tail is not None and tail != root.name:
                yield (depth, ch, grad)
                continue

            is_parameter = id(getattr(ch, 'variable', None)) in self.params
            constraint = 'false' if is_parameter and not self.flat else None
            self.dot.edge(id_(ch), id_(grad), constraint=constraint)

        for var in getattr(grad, 'saved_tensors', ()) or ():
            if torch.is_tensor(var):
                root.node(id_(var), f'{tuple(var.shape)}', fillcolor='orange')
                self.dot.edge(id_(var), id_(grad))

    def _mark(self, ts):
        edges = []
        for t in as_tuple(ts):
            if t.grad_fn is not None:
                self._shapes[t.grad_fn] = t.shape
                edges.extend(self._traverse(t.grad_fn, None))
        if not edges:
            return

        max_depth = max(depth for depth, *_ in edges) + 1
        for depth, tail, head in edges:
            if tail is not None:
                minlen = None if self.flat else f'{max_depth - depth}'
                self.dot.edge(id_(tail), id_(head), minlen=minlen)

    def forward_pre(self, name, module, xs):
        self._mark(xs)
        # -------- start node --------
        if self.flat:
            return
        scope = graphviz.Digraph(name=f'cluster_{self._mangle(name)}')
        scope.attr(label=f'{name.split(".")[-1]}:{type(module).__name__}')
        self._stack.append(scope)

    def forward(self, module, _, ys):
        self._mark(ys)
        if self.flat:
            return
        self._stack[-2].subgraph(self._stack.pop(-1))
        # -------- end node --------


def plot_model(model: torch.nn.Module, *input_shapes: tuple, device='cpu'):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    hk = Builder(
        [
            torch.zeros(1, *shape, device=device, requires_grad=True)
            for shape in input_shapes
        ],
        {
            id(v): 'root{}'.format(f'.{name}' if name else '')
            for name, v in model.state_dict(keep_vars=True).items()
        },
    )
    with ExitStack() as stack:
        for name, m in model.named_modules():
            name = 'root{}'.format(f'.{name}' if name else '')
            for handle in (
                m.register_forward_pre_hook(partial(hk.forward_pre, name)),
                m.register_forward_hook(hk.forward),
            ):
                stack.callback(handle.remove)
        model(*hk.inputs)

    dot = hk.dot

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
