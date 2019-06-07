__all__ = 'get_gpu_state', 'plot_model'

import os

import torch
from graphviz import Digraph


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
    return (limit // 2 ** 20), len(ids)


def addr(x):
    return f'0x{id(x):x}'


def plot_model(model: torch.nn.Module, *input_shapes: tuple, device='cpu'):
    """Produces Graphviz representation of PyTorch autograd graph

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
            'height': '0.2',
        },
        graph_attr={'size': '12,12'},
        format='svg',
    )

    seen = set()
    # batch_size = 2, otherwise BatchNorm will fail
    inputs = (torch.randn(2, *shape, device=device) for shape in input_shapes)
    var = model(*inputs)
    if not isinstance(var, tuple):
        var = (var, )

    params = {addr(v): k for k, v in model.state_dict(keep_vars=True).items()}
    outputs = tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var in seen:
            return
        seen.add(var)

        if torch.is_tensor(var):
            dot.node(addr(var), tuple(var.shape), fillcolor='orange')
        elif hasattr(var, 'variable'):
            u = var.variable
            dot.node(addr(var), f'{params.get(addr(u))}\n {tuple(u.shape)}',
                     fillcolor='lightblue')
        elif var in outputs:
            dot.node(addr(var), type(var).__name__.replace('Backward', ''),
                     fillcolor='darkolivegreen1')
        else:
            dot.node(addr(var), type(var).__name__.replace('Backward', ''))

        if hasattr(var, 'next_functions'):
            for u, *_ in var.next_functions:
                if u is not None:
                    dot.edge(addr(var), addr(u))
                    add_nodes(u)
        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                dot.edge(addr(var), addr(t))
                add_nodes(t)

    for v in outputs:
        add_nodes(v)

    size_max = 12
    scale_factor = .15
    size = max(size_max, len(dot.body) * scale_factor)
    dot.graph_attr.update(size=f'{size},{size}')

    os.makedirs('graphs', exist_ok=True)
    try:
        model_name = model.name
    except AttributeError:
        model_name = type(model).__name__
    with open(f'graphs/{model_name}.svg', 'wb') as f:
        f.write(dot.pipe())
