__all__ = 'get_gpu_state', 'plot_model'

import functools
import os
from contextlib import ExitStack

import graphviz
import torch

from ..core import decimate


def get_gpu_state():
    from py3nvml.py3nvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )
    nvmlInit()
    try:
        ids = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        ids = list(range(nvmlDeviceGetCount()))
    else:
        ids = [int(device_id) for device_id in ids.split(',')]
    devices = (nvmlDeviceGetHandleByIndex(i) for i in ids)
    limit = sum(nvmlDeviceGetMemoryInfo(dev).free for dev in devices)
    nvmlShutdown()
    return (limit // 2**20), len(ids)


def id_(x):
    return hex(id(x))


def as_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )


def named(typed):
    return type(typed).__name__.replace('Backward', '')


def sized(tensor):
    return f'{decimate(tensor.nelement() * tensor.element_size())}B'


def hook(shapes: dict, m, *ts):
    for t in ts:
        shapes.update({v.grad_fn: tuple(v.shape) for v in as_tuple(t)})


def plot_model(model: torch.nn.Module, *input_shapes: tuple, device='cpu'):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    dot = graphviz.Digraph(
        filename=getattr(model, 'name', type(model).__name__),
        directory='graphs',
        format='svg',
        node_attr={
            'style': 'filled',
            'shape': 'box',
            'align': 'left',
            'fontsize': '12',
            'ranksep': '0.1',
            'height': '0.2',
        },
    )

    inputs = [torch.zeros(2, *shape, device=device, requires_grad=True)
              for shape in input_shapes]
    shapes = {}
    with ExitStack() as stack:
        model.apply(lambda m: stack.callback(
            m.register_forward_hook(functools.partial(hook, shapes)).remove
            ))
        var = model(*inputs)

    var = as_tuple(var)
    params = {id(v): k for k, v in model.state_dict(keep_vars=True).items()}
    outputs = tuple(v.grad_fn for v in var)

    @functools.lru_cache(maxsize=None)  # call once for each unique `var`
    def add_nodes(var):
        if torch.is_tensor(var):
            dot.node(id_(var), tuple(var.shape), fillcolor='orange')
        elif hasattr(var, 'variable'):
            u = var.variable
            dot.node(
                id_(var),
                f'{params.get(id(u))} : {tuple(u.shape)} : {sized(u)}',
                fillcolor=('lightblue', 'orange')[id(u) in map(id, inputs)]
            )
        else:
            dot.node(
                id_(var), named(var),
                fillcolor=('lightgrey', 'darkolivegreen1')[var in outputs]
            )

        for gen in (
            (c for c, *_ in getattr(var, 'next_functions', ())),
            getattr(var, 'saved_tensors', ()),
        ):
            for child in gen:
                if child is None:
                    continue
                dot.edge(id_(child), id_(var),
                         label=f'{shapes[child]}' if child in shapes else None)
                add_nodes(child)

    for v in outputs:
        add_nodes(v)

    size_min = 12
    scale_factor = .15
    size = max(size_min, len(dot.body) * scale_factor)
    dot.graph_attr.update(size=f'{size},{size}')
    dot.render(cleanup=True)
    return dot
