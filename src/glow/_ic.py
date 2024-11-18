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
from typing import Any, NamedTuple, overload

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
    arr = arr.squeeze()
    hs, ms = np.divmod(arr.shape, 2)

    # Pyramid of splits
    splits: dict[tuple[int, ...], np.ndarray] = {(): arr.astype('f')}
    for axis, (half, m) in enumerate(zip(hs, ms)):
        sep = [half, half + 1] if m else [half]
        splits = {
            (*k, k2): ss
            for k, s in splits.items()
            for k2, ss in enumerate(np.split(s, sep, axis))
        }

    # Tensor of means, (low, 0?, high) ^ ndim
    s_shape = 2 * hs.clip(max=1) + ms
    sums = np.zeros(s_shape)
    counts = np.zeros(s_shape, 'i')
    for loc, s in splits.items():
        n = s.size
        if (mask := np.ma.getmask(s)) is not np.ma.nomask:
            n -= mask.sum()
        if n:
            sums[loc] = s.sum()
        counts[loc] = n

    # Aggregate and do grads
    hsums = np.empty((arr.ndim, 2), 'f')
    for axis in range(arr.ndim):
        axes = *range(axis), *range(axis + 1, arr.ndim)
        hsums[axis] = (sums.sum(axes) / counts.sum(axes).clip(min=1))[[0, -1]]
    return hsums @ [-1, 1]


_MAX_SIZES = {'b': 40, 'i': 40, 'u': 40, 'f': 20, 'c': 20}


def _fmt_1d(a: np.ndarray, tol: int = 4) -> str:
    if a.size == 1:
        return f'{a.item():.{tol}g}'
    return '[' + ' '.join(f'{x:.{tol}g}' for x in a.tolist()) + ']'


def _bool_info(arr: np.ndarray) -> Iterator[str]:
    if arr.size < 500:  # all data in Packed Hex
        line = np.packbits(arr.flat).tobytes().hex().replace('0', '_')
        yield f'bits={line!r}'

    else:  # histrogram + gradient
        weights = np.bincount(arr.ravel()).astype('d') / arr.size
        yield f'bool @ {_fmt_1d(weights)}'
        yield from _grad_info(arr)


def _grad_info(arr: np.ndarray | np.ma.MaskedArray) -> Iterator[str]:
    if (grad := _get_nd_grad(arr)).any():
        yield f'grad={_fmt_1d(grad)}'


def _get_properties(arr: np.ndarray, lo, hi) -> Iterator[str]:
    dtype: np.dtype = arr.dtype
    assert arr.size
    match dtype.kind:
        case 'b':  # Bool
            yield from _bool_info(arr)

        case 'u' | 'i':  # Integers
            range_ = int(hi) - int(lo) + 1

            # Small range (lo >= 0 and hi <= 10), show distribution
            if lo >= 0 and hi <= 10:
                uniq, counts = np.unique(arr, return_counts=True)
                weights = counts.astype('d') / arr.size

                if range_ == uniq.size:  # 100% range, no skips
                    yield f'{dtype}∈[{lo} ... {hi}] @ {_fmt_1d(weights)}'
                else:
                    yield f'{dtype}∈{uniq} @ {_fmt_1d(weights)}'

            else:  # Wide range - low/high + mean/std + nuniq (opt) + gradient
                yield f'{dtype}({arr.mean():.3g} ± {arr.std():.3g})'
                yield f'X∈[{lo} ... {hi}]'

                # Not much uniqs
                if (
                    range_ < 1_000 or arr.size < 1_000_000 or arr.itemsize <= 2
                ) and (nuniq := np.unique(arr).size) != range_:
                    yield f'{nuniq / range_:.2%} range'

            yield from _grad_info(arr)

        case 'c' | 'f':  # Dense data, use mean/std/gradient

            if dtype.kind == 'c':  # Force complex as float
                arr = arr.astype('F').view('2f')
                lo, hi = arr.min(), arr.max()  # Complex min/max uses amplitude

            arr = np.ma.masked_invalid(arr)
            mask = np.ma.getmask(arr)
            if (num_invalid := mask.sum()) < arr.size:
                if num_invalid:  # Old min/max have invalid data, recompute
                    lo, hi = arr.min(), arr.max()
                    yield f'{num_invalid / arr.size:.2%} invalid'

                if lo < hi:
                    yield f'{dtype}({arr.mean():.3g} ± {arr.std():.3g})'
                    yield f'X∈[{lo:.3g} ... {hi:.3g}]'
                else:
                    yield f'{dtype}({lo:.3g})'

            # NaN/-Inf/+Inf
            yield from map(str, np.unique(arr.data[mask]).tolist())

            yield from _grad_info(arr)

        case _:
            raise NotImplementedError(f'Unknown dtype: {dtype}')


class _ReprArray(NamedTuple):
    data: np.ndarray

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        arr = self.data
        if not arr.size:
            return f'{arr!r}'  # ndarray.__repr__ is short for 0-sized arrays
        dtype = arr.dtype

        # Single value
        if arr.size == 1:
            return f'np.{dtype}({arr})'

        # Try to collapse
        shape = f'{arr.shape if arr.ndim != 1 else arr.size}'
        if dtype.kind in 'buifc':
            lo, hi = arr.min(), arr.max()
            if np.isfinite([lo, hi]).all() and lo == hi:  # "full" array
                match lo:
                    case 0:
                        template = 'np.zeros({shape}, {dtype})'
                    case 1:
                        template = 'np.ones({shape}, {dtype})'
                    case _:
                        template = 'np.full({shape}, {dtype}({value}))'
                return template.format(shape=shape, dtype=dtype, value=lo)

        # Small array (or unknown dtype), print contents as is
        if arr.size < _MAX_SIZES.get(dtype.kind, np.inf):
            if arr.ndim == 1 or arr.size == 1:
                return f'np.{dtype}({arr})'
            return f'np.array({shape}, {dtype}({arr.ravel()}))'

        return (
            'np.array('
            + ', '.join([shape, *_get_properties(arr, lo, hi)])
            + ')'
        )


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

    return (
        basename(info.filename)
        + ('' if call_node is None else f':{call_node.lineno}')
        + f' in {parent_fn}'
    )


def _construct_argument_output(
    context: str, pairs: Iterable[tuple[str, Any]]
) -> str:
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
        val if is_literal(arg) else f'{arg}: {val}' for arg, val in pairs
    )

    if len(single_line_formatted_args.splitlines()) <= 1:
        all_pairs = (
            f'{PREFIX}{context} - {single_line_formatted_args}'
            if context
            else f'{PREFIX}{single_line_formatted_args}'
        )
        if len(all_pairs.splitlines()[0]) <= LINE_WRAP_WIDTH:
            # ic| foo.py:11 in foo() - a: 1, b: 2
            # ic| a: 1, b: 2, c: 3
            return all_pairs

    lines = tuple(format_pair(arg, value) for arg, value in pairs)
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
def ic() -> None: ...


@overload
def ic[T](x: T, /) -> T: ...


@overload
def ic[*Ts](*xs: *Ts) -> tuple[*Ts]: ...


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
