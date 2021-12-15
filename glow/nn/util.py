from __future__ import annotations

__all__ = [
    'device', 'dump_to_onnx', 'frozen', 'inference', 'max_tune', 'param_count',
    'profile'
]
import functools
import pickle
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Callable, TypeVar, cast

import torch
import torch.autograd
import torch.cuda
import torch.jit
import torch.onnx
from torch import jit, nn

from .. import si

try:
    if sys.platform == 'win32':  # Not available on Windows
        raise ImportError
    import tensorrt as trt
    import torch_tensorrt as pt_trt
except ImportError:
    trt = pt_trt = None

_F = TypeVar('_F', bound=Callable[..., Iterator])
_ModuleLike = Callable[[torch.Tensor], torch.Tensor]


def device() -> torch.device:
    """Gets current device, including CPU"""
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device('cpu')


@contextmanager
def _set_eval(net: nn.Module) -> Iterator[None]:
    """Locally switch net to eval-mode"""
    was_train = net.training
    try:
        net.eval()
        yield
    finally:
        net.train(was_train)


@contextmanager
def frozen(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while training.

    Net is switched to eval mode, and its parameters are detached from graph.
    Grads are computed if inputs require grad.
    Works as context manager"""
    with ExitStack() as stack:
        stack.enter_context(_set_eval(net))
        for p in net.parameters():
            if p.requires_grad:
                stack.callback(p.requires_grad_)
                p.detach_()
        yield


@contextmanager
def inference(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while inference.

    Net is switched to eval mode, and gradient computation is turned off.
    Works as context manager"""
    with _set_eval(net), torch.inference_mode():
        yield


def param_count(net: nn.Module) -> int:
    """Count of parameters in net, both training and not"""
    return si(sum(p.numel() for p in net.parameters()))


def profile(fn: _F) -> _F:
    """Decorator to profile CUDA ops. Use with `nvprof`

    Use in script launched via:
    ```bash
    nvprof --profile-from-start off -o trace.prof -- python main.py
    ```
    Usage:
    >>> @profile
    ... def train_loop():
    ...     for data in loader:
    ...         yield step(data)

    """
    def wrapper(*args, **kwargs):
        results = fn(*args, **kwargs)
        with torch.cuda.profiler.profile():
            yield from islice(results, 1)
            with torch.autograd.profiler.emit_nvtx():
                yield from results

    return cast(_F, functools.update_wrapper(wrapper, fn))


def dump_to_onnx(net: nn.Module,
                 *shapes: tuple[int, ...],
                 device: str = 'cpu') -> bytes:
    """Converts model to ONNX graph, represented as bytes

    Parameters:
    - net - torch.nn.Module to convert
    - shapes - Shapes of input data, all except batch dimension

    Example usage:
    >>> net = torch.nn.Linear(4, 4)
    >>> bytes_ = dump_to_onnx(net, [4])

    To restore graph:
    >>> from onnxruntime import backend
    >>> rep = backend.prepare(bytes_or_filename, device='cpu')
    >>> rep.run([np.zeros(4, 4)])[0]

    """
    dynamic_axes = {
        f'inp_{i}': {
            0: 'batch',
            **{dim: f'inp_{i}_dim_{dim}' for dim in range(2, 1 + len(shape))}
        } for i, shape in enumerate(shapes)
    }
    buf = BytesIO()
    torch.onnx.export(
        net.to(device).eval(),
        tuple(
            torch.rand(1, *shape, requires_grad=True, device=device)
            for shape in shapes),
        buf,
        input_names=[*dynamic_axes],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True)
    return buf.getvalue()


# ----------------- tracing ----------------------


class LazilyTraced(nn.Module):
    def __init__(self, impl: nn.Module):
        super().__init__()
        self.impl = impl
        self.traced: torch.jit.TracedModule | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.impl(x)

        if self.traced is None:
            with torch.autocast('cuda', False):
                self.traced = torch.jit.trace(self.impl, x[:2])
        assert self.traced is not None
        return self.traced(x)

    def save(self, path: Path, **metadata):
        buf = BytesIO()
        if self.traced is not None:
            self.traced.save(buf)
        with path.open('wb') as fp:
            pickle.dump({'traced': buf.getvalue(), 'meta': metadata}, fp)


# ---------------------------- tune for inference ----------------------------


@torch.inference_mode()
def _bench(fn, *data, n=50):
    for _ in range(3):
        fn(*data)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    for _ in range(n):
        fn(*data)
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f'Batches/s: {n / (t2 - t1):.2f}')


class _WithHalf(nn.Sequential):
    def forward(self, *args):
        args = (a.half() for a in args)
        return super().forward(*args)
        # if isinstance(args, tuple):
        #     return *(a.float() for a in args),
        # return args.float()


class _WithAutocast(nn.Module):
    def __init__(self, base: nn.Module, device: torch.device):
        super().__init__()
        self.base = base
        self.dev_type = device.type

    def forward(self, x):
        with torch.autocast(self.dev_type), torch.inference_mode():
            return self.base(x)


def _convert_to_pt_trt(mod: nn.Module,
                       *static_inputs: torch.Tensor,
                       fp16: bool = False,
                       path: Path | None = None) -> nn.Module:
    assert pt_trt
    if path and path.exists():
        return jit.load(path)

    inputs = *[
        pt_trt.Input(
            min_shape=[1, *i.shape[1:]],
            opt_shape=i.shape,
            max_shape=i.shape,
            dtype=i.dtype,
        ) for i in static_inputs
    ],
    precisions = {torch.float}
    if fp16:
        precisions = {torch.half}

    mod = pt_trt.compile(
        mod,
        inputs=inputs,
        enabled_precisions=precisions,
        # truncate_long_and_double=True,
        # torch_executed_ops=['aten::upsample_bilinear2d'],
        # workspace_size=1 << 24,
        # require_full_compilation=True,
    )
    if path:
        jit.save(mod, path)
    return mod


class _WithArgs(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, *args):
        kwargs = {f'inp_{i}': a for i, a in enumerate(args)}
        res = *self.base(kwargs).values(),
        if len(res) == 1:
            return res[0]
        return res


def _convert_to_trt(mod: nn.Module,
                    *static_inputs: torch.Tensor,
                    fp16: bool = False,
                    path: Path | None = None) -> nn.Module:
    assert trt
    from io import BytesIO

    import onnx
    import torch.onnx
    from mmcv.tensorrt import (TRTWrapper, load_trt_engine, onnx2trt,
                               save_trt_engine)

    if path and path.exists():
        return _WithArgs(TRTWrapper(load_trt_engine(path)))

    dynamic_axes = {
        f'inp_{i}': {
            0: 'batch'
        } for i, _ in enumerate(static_inputs)
    }
    fp = BytesIO()
    torch.onnx.export(
        mod,
        static_inputs,
        fp,
        opset_version=13,
        input_names=[*dynamic_axes],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,  # ?
        keep_initializers_as_inputs=True,  # ?
    )

    fp.seek(0)
    engine = onnx2trt(
        onnx.load(fp), {
            i: [[1, *s.shape[1:]], s.shape, s.shape]
            for i, s in zip(dynamic_axes, static_inputs)
        },
        fp16_mode=fp16)
    if path:
        save_trt_engine(engine, path)

    return _WithArgs(TRTWrapper(engine))


def _capture_cuda_graph(mod: nn.Module,
                        *static_inputs: torch.Tensor) -> _ModuleLike:
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            mod(*static_inputs)
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outputs = mod(*static_inputs)

    def wrapper(*args: torch.Tensor):
        for src, dst in zip(args, static_inputs):
            if src.shape != dst.shape:
                return mod(*args)  # Fallback on shape mismatch
            dst.copy_(src)
        g.replay()
        return outputs

    return wrapper


def tune_for_inference(mod: nn.Module,
                       *shapes: tuple[int, ...],
                       fp16: bool,
                       device: torch.device,
                       use_trt: bool = False,
                       use_graph: bool = False) -> nn.Module | _ModuleLike:
    use_trt = bool(trt or pt_trt) and use_trt
    inputs = *(torch.rand(shape, device=device) for shape in shapes),

    if fp16:
        mod.requires_grad_(False)

    if use_trt:
        # try:
        #     jit.script(mod)  # Python API -> TorchScript
        # except Exception as exc:  # noqa: PIE786
        #     print(f'JIT failed with: {exc}')
        #     return mod

        if fp16:
            inputs = *(i.half() for i in inputs),
            mod.half()

        if trt:
            mod = _convert_to_trt(mod, *inputs, fp16=fp16)
        else:
            mod = jit.trace(mod, inputs)  # Python API -> TorchScript
            mod = jit.freeze(mod)  # Fold constants
            mod = _convert_to_pt_trt(mod, *inputs, fp16=fp16)

        return _WithHalf(mod) if fp16 else mod

    else:
        if fp16:
            mod = _WithAutocast(mod, device).train(mod.training)

        mod = jit.trace(mod, inputs)  # Python API -> TorchScript
        mod = jit.freeze(mod)  # Fold constants
        mod = jit.optimize_for_inference(mod)  # Optimize kernels

        with open('graph.txt', 'w') as fp:  # ! debug
            print(mod.inlined_graph, file=fp)

        # Eager execution to static graph (reduces CPU load)
        if use_graph:
            return _capture_cuda_graph(mod, *inputs)
        return mod


def max_tune(mod: nn.Module,
             shape: tuple[int, ...],
             fp16: bool,
             device: torch.device,
             depth: int = 0) -> nn.Module:

    tune = partial(tune_for_inference, fp16=fp16, device=device, use_trt=True)
    if not depth:
        return tune(mod, shape)  # type: ignore

    ishapes: dict[str, list[tuple]] = {}

    def hook(name: str, _, xs: tuple[torch.Tensor, ...]):
        ishapes[name] = [x.shape for x in xs]

    depth -= 1
    mods = {
        name: m for name, m in mod.named_modules()
        if (name.count('.') == depth and m is not mod and
            not isinstance(m, nn.Identity))
    }
    with ExitStack() as s:
        for name, m in mod.named_modules():
            s.callback(m.register_forward_pre_hook(partial(hook, name)).remove)
            s.callback(m.train, m.training)

        with torch.inference_mode(), torch.autocast(device.type, fp16):
            mod.eval()
            mod(torch.zeros(shape, device=device))

    for name, m in mods.items():
        if name not in ishapes:  # inactive module
            continue
        *heads, tail = name.split('.')
        print(name, type(m).__name__)

        if heads:
            parent = mod.get_submodule('.'.join(heads))
            setattr(parent, tail, tune(m, *ishapes[name]))
        else:
            setattr(mod, tail, tune(m, *ishapes[name]))

    return mod
