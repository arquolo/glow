import os

import torch as T
from graphviz import Digraph

from ..tool import export


@export
def get_gpu_state():
    from py3nvml.py3nvml import (nvmlInit, nvmlShutdown,
                                 nvmlDeviceGetCount,
                                 nvmlDeviceGetHandleByIndex,
                                 nvmlDeviceGetMemoryInfo)
    nvmlInit()
    ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    ids = ([int(device_id) for device_id in ids.split(',')] if ids
           else list(range(nvmlDeviceGetCount())))
    limit = sum(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
                for i in ids) // 1024 ** 2
    nvmlShutdown()
    return limit, len(ids)


@export
def plot_model(model: T.nn.Module, input_shape: tuple, device='cpu'):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    dot = Digraph(
        node_attr={
            'style': 'filled',
            'shape': 'box',
            'align': 'left',
            'fontsize': '12',
            'ranksep': '0.1',
            'height': '0.2'
        },
        graph_attr={
            'size': '12,12'
        },
        format='svg')

    seen = set()
    var = model(T.randn(2, *input_shape, device=device))  # 2 for BatchNorm

    def addr(x):
        return f'{hex(id(x))}'

    params = model.state_dict(keep_vars=True)
    param_map = {addr(v): k for k, v in params.items()}
    output_nodes = (tuple(v.grad_fn for v in var) if isinstance(var, tuple)
                    else (var.grad_fn,))

    def add_nodes(var):
        if var in seen:
            return
        seen.add(var)

        if T.is_tensor(var):
            dot.node(addr(var), tuple(var.shape), fillcolor='orange')
        elif hasattr(var, 'variable'):
            u = var.variable
            dot.node(addr(var), f'{param_map.get(addr(u))}\n {tuple(u.shape)}',
                     fillcolor='lightblue')
        elif var in output_nodes:
            dot.node(addr(var), type(var).__name__.replace('Backward', ''),
                     fillcolor='darkolivegreen1')
        else:
            dot.node(addr(var), type(var).__name__.replace('Backward', ''))

        if hasattr(var, 'next_functions'):
            for u, *_ in var.next_functions:
                if u is not None:
                    dot.edge(addr(u), addr(var))
                    add_nodes(u)
        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                dot.edge(addr(t), addr(var))
                add_nodes(t)

    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    size = max(12, len(dot.body) * .15)
    dot.graph_attr.update(size=f'{size},{size}')

    os.makedirs('graphs', exist_ok=True)
    try:
        model_name = model.name
    except AttributeError:
        model_name = type(model).__name__
    with open(f'graphs/{model_name}.svg', 'wb') as f:
        f.write(dot.pipe())
