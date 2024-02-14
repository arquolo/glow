"""
IceCream - Never use print() to debug again

Ansgar Grunseid
grunseid.com
grunseid@gmail.com

Pavel Maevskikh
arquolo@gmail.com

License: MIT

pip install asttokens colorama executing numpy pygments
"""

__all__ = ['ic', 'ic_repr']

import ast
import inspect
import pprint
import shutil
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from os.path import basename
from textwrap import dedent
from threading import Lock
from types import FrameType
from typing import Any, NamedTuple, TypeVar, TypeVarTuple, Unpack, overload

import colorama
import executing
import numpy as np
from executing.executing import EnhancedAST
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers.python import PythonLexer

colorama.init()
LOCK = Lock()

PREFIX = 'ic| '
LINE_WRAP_WIDTH = 70  # Characters

FORMATTER = TerminalFormatter(bg='dark')
LEXER = PythonLexer(ensurenl=False)

_T = TypeVar('_T')
_Ts = TypeVarTuple('_Ts')


def is_literal(s) -> bool:
    try:
        ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


class Source(executing.Source):
    def get_text_with_indentation(self, node) -> str:
        result = self.asttokens().get_text(node)
        if '\n' in result:
            result = ' ' * node.first_token.start[1] + result
            result = dedent(result)
        return result.strip()


def indented_lines(prefix: str, lines: str) -> list[str]:
    space = ' ' * len(prefix)
    first, *rest = lines.splitlines()
    return [prefix + first] + [space + line for line in rest]


def format_pair(arg: str, value: str) -> str:
    # Align the start of multiline strings.
    if value[0] + value[-1] in ["''", '""']:
        value = ' '.join(value.splitlines(keepends=True))

    *lines, tail = arg.splitlines()
    return '\n'.join(lines + indented_lines(tail + ': ', value))


def _get_nd_grad(arr: np.ndarray) -> np.ndarray:
    # A bit sophisticated way to compute gradients by all directions,
    # but a fastest one.
    # Split tensor by all axes, do mean for each cell,
    # and then aggregate means to mean for the each axis split.
    arr_f4 = arr.astype('f4')

    # Pyramid of splits
    splits: dict[tuple[int, ...], np.ndarray] = {(): arr_f4}
    for axis, size in enumerate(arr.shape):
        if size == 1:
            splits = {(*k, 0): s for k, s in splits.items()}
        else:
            half = size // 2
            splits = {
                (*k, k2): ss for k, s in splits.items()
                for k2, ss in enumerate(np.split(s, [half, -half], axis))
            }

    # Tensor of means, (low, 0, high) ^ ndim
    s_shape = [1 if size == 1 else 3 for size in arr.shape]
    means = np.zeros(s_shape)
    weights = np.zeros(s_shape, int)
    for loc, s in splits.items():
        means[loc] = s.mean() if s.size else 0
        weights[loc] = s.size

    # Aggregate and do grads
    grad = np.zeros(arr.ndim)
    for axis, size in enumerate(arr.shape):
        if size == 1:
            continue

        axes = *(a for a in range(arr.ndim) if a != axis),
        means_ = np.take(means, [0, 2], axis)
        weights_ = np.take(weights, [0, 2], axis)
        head, tail = np.average(means_, axes, weights_)
        grad[axis] = tail - head

    return grad


def _get_properties(arr: np.ndarray) -> Iterator[str]:
    yield f'{arr.shape}, {arr.dtype}'
    if not arr.size:
        return

    # Small array, print contents as is
    if arr.size < {'b': 40, 'u': 40, 'i': 40}.get(arr.dtype.kind, 20):
        yield f'data={arr.copy().ravel()}'
        return

    # Small enough binary array, hexify
    if arr.size < 500 and arr.dtype == bool:
        data = np.packbits(arr.flat).tobytes()
        line = ''.join(f'{v:02x}' for v in data).replace('0', '_')
        yield f'data={line!r}'
        return

    # Too much data, use statistics
    lo = arr.min()
    hi = arr.max()

    if arr.dtype.kind == 'f':
        yield f'x∈[{lo:.8f}, {hi:.8f}]'
        yield f'x={arr.mean():.8f}±{arr.std():.8f}'

    # Wide range, only low/high
    elif int(hi) - int(lo) > 100:
        yield f'x∈[{lo}, {hi}]'

    # Medium range or zero crossing, low/high + nuniq
    elif int(lo) < 0 or int(hi) > 10:
        nuniq = np.unique(arr).size
        yield f'x∈[{lo}, {hi}], nuniq={nuniq}'

    # Narrow range, raw distribution
    else:
        weights = np.bincount(arr.flat).astype('f8') / arr.size
        yield f'weights={weights}'

    if (grad := _get_nd_grad(arr)).any():
        yield f'grad={grad.round(8)}'


class _ReprArray(NamedTuple):
    data: np.ndarray

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return 'np.ndarray(' + ', '.join(_get_properties(self.data)) + ')'


def _prepare(obj):  # noqa: PLR0911
    if isinstance(obj, np.ndarray):
        return _ReprArray(obj)

    if isinstance(obj, str | bytes | bytearray | range | Iterator):
        return obj

    if is_dataclass(obj):
        items = {f.name: getattr(obj, f.name) for f in fields(obj) if f.init}
        return replace(obj, **_prepare(items))  # type: ignore

    # namedtuple
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*map(_prepare, obj))

    if isinstance(obj, Mapping):
        return {_prepare(k): _prepare(v) for k, v in obj.items()}

    if isinstance(obj, Iterable):
        return type(obj)(map(_prepare, obj))  # type: ignore[call-arg]

    return obj


def ic_repr(obj: Any, width: int | None = None) -> str:
    obj_repr = _prepare(obj)

    if width is None:
        width = shutil.get_terminal_size().columns
    line = pprint.pformat(obj_repr, width=width)

    # Preserve string newlines in output.
    return line.replace('\\n', '\n')


def _format_time() -> str:
    now = f'{datetime.now():%H:%M:%S.%f}'[:-3]  # Tail is not usecs, but msecs
    return f' at {now}'


def _format_context(frame: FrameType, call_node: EnhancedAST | None) -> str:
    info = inspect.getframeinfo(frame)
    parent_fn = info.function

    if parent_fn != '<module>':
        parent_fn = f'{parent_fn}()'

    return (basename(info.filename) +
            ('' if call_node is None else f':{call_node.lineno}')
            + f' in {parent_fn}')


def _construct_argument_output(context: str,
                               pairs: Iterable[tuple[str, Any]]) -> str:
    pairs = [(arg, ic_repr(val)) for arg, val in pairs]
    # For cleaner output, if <arg> is a literal, eg 3, "string", b'bytes',
    # etc, only output the value, not the argument and the value, as the
    # argument and the value will be identical or nigh identical. Ex: with
    # ic("hello"), just output
    #
    #   ic| 'hello',
    #
    # instead of
    #
    #   ic| "hello": 'hello'.
    #
    single_line_formatted_args = ', '.join(
        val if is_literal(arg) else f'{arg}: {val}' for arg, val in pairs)

    if len(single_line_formatted_args.splitlines()) <= 1:
        all_pairs = (f'{PREFIX}{context} - {single_line_formatted_args}'
                     if context else f'{PREFIX}{single_line_formatted_args}')
        if len(all_pairs.splitlines()[0]) <= LINE_WRAP_WIDTH:
            # ic| foo.py:11 in foo() - a: 1, b: 2
            # ic| a: 1, b: 2, c: 3
            return all_pairs

    lines = *(format_pair(arg, value) for arg, value in pairs),
    if context:
        # ic| foo.py:11 in foo()
        #     multilineStr: 'line1
        #                    line2'
        #
        # ic| foo.py:11 in foo()
        #     a: 11111111111111111111
        #     b: 22222222222222222222
        lines = context, *lines
    else:
        # ic| multilineStr: 'line1
        #                    line2'
        #
        # ic| a: 11111111111111111111
        #     b: 22222222222222222222
        pass
    return '\n'.join(indented_lines(PREFIX, '\n'.join(lines)))


def _format(frame: FrameType, *args) -> str:
    call_node = Source.executing(frame).node
    context = _format_context(frame, call_node)
    if not args:
        return PREFIX + context + _format_time()

    if call_node is not None:
        source: Source = Source.for_frame(frame)  # type: ignore[assignment]
        sanitized_arg_strs = [
            source.get_text_with_indentation(arg)
            for arg in call_node.args  # type: ignore[attr-defined]
        ]
        pairs = zip(sanitized_arg_strs, args)
    else:
        pairs = ((f'{i}', arg) for i, arg in enumerate(args))

    return _construct_argument_output(context, pairs)


@overload
def ic() -> None:
    ...


@overload
def ic(x: _T) -> _T:
    ...


@overload
def ic(*xs: Unpack[_Ts]) -> tuple[*_Ts]:
    ...


def ic(*args):
    frame = inspect.currentframe()
    assert frame
    assert frame.f_back
    out = _format(frame.f_back, *args)

    s = highlight(out, LEXER, FORMATTER)
    with LOCK:
        print(s, file=sys.stderr)

    if not args:
        return None  # E.g. ic().
    if len(args) == 1:
        return args[0]  # E.g. ic(1).
    return args  # E.g. ic(1, 2, 3).
