__all__ = ['lock_seed', 'trace', 'trace_module', 'whereami']

import gc
import os
import random
from collections.abc import Callable, Iterator
from contextlib import suppress
from inspect import currentframe, getmodule, isfunction
from itertools import islice
from types import FrameType, ModuleType

import numpy as np
import wrapt

from ._cache import memoize
from ._import_hook import register_post_import_hook


def _frame_hash(frame: FrameType) -> tuple[str, int]:
    return frame.f_code.co_filename, frame.f_lineno


@memoize(100, policy='lru', key_fn=_frame_hash)
def _get_source(frame: FrameType) -> str:
    # Get source module name
    modname = (
        spec.name
        if (module := getmodule(frame)) and (spec := module.__spec__)
        else '__main__'
    )

    # Get source code name (method or function name)
    code = frame.f_code
    codename = next(
        (f.__qualname__ for f in gc.get_referrers(code) if isfunction(f)),
        code.co_name,
    )
    if codename == '<module>':
        codename = ''

    return f'{modname}:{codename}:{frame.f_lineno}'


def _get_source_calls(frame: FrameType | None) -> Iterator[str]:
    while frame:
        yield _get_source(frame)
        if frame.f_code.co_name == '<module>':  # Stop on module-level scope
            return
        frame = frame.f_back


def stack(skip: int = 0, limit: int | None = None) -> Iterator[str]:
    """Return iterator of FrameInfos, stopping on module-level scope."""
    frame = currentframe()
    calls = _get_source_calls(frame)
    calls = islice(calls, skip + 1, None)  # Skip 'skip' outerless frames
    if not limit:
        return calls
    return islice(calls, limit)  # Keep at most `limit` outer frames


def whereami(skip: int = 0, limit: int | None = None) -> str:
    calls = stack(skip + 1, limit)
    return ' -> '.join(reversed([*calls]))


@wrapt.decorator
def trace(fn, _, args, kwargs):
    print(
        f'<({whereami(3)})> : {fn.__module__ or ""}.{fn.__qualname__}',
        flush=True,
    )
    return fn(*args, **kwargs)


def _set_trace(
    obj: ModuleType | Callable,
    *,
    seen: set[str] | None = None,
    prefix: str | None = None,
    module: ModuleType | None = None,
) -> None:
    # TODO: rewrite using unittest.mock
    if isinstance(obj, ModuleType):
        if seen is None:
            seen = set()
            prefix = obj.__name__
        assert isinstance(prefix, str)
        if not obj.__name__.startswith(prefix) or obj.__name__ in seen:
            return
        seen.add(obj.__name__)
        for name in dir(obj):
            _set_trace(
                getattr(obj, name), module=obj, seen=seen, prefix=prefix
            )

    if not callable(obj):
        return

    assert isinstance(module, ModuleType)
    if not hasattr(obj, '__dict__'):
        setattr(module, obj.__qualname__, trace(obj))
        print(f'wraps "{module.__name__}:{obj.__qualname__}"')
        return

    for name in obj.__dict__:
        with suppress(AttributeError, TypeError):
            member = getattr(obj, name)
            if not callable(member):
                continue
            decorated = trace(member)

            for m in (decorated, member, obj):
                with suppress(AttributeError):
                    decorated.__module__ = m.__module__
                    break
            else:  # noqa: PLW0120, RUF100
                decorated.__module__ = getattr(module, '__name__', '')
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


def trace_module(name: str) -> None:
    """Enable call logging for each callable inside module name."""
    register_post_import_hook(_set_trace, name)


# ---------------------------------------------------------------------------


def lock_seed(seed: int) -> None:
    """Set seed for all modules: random/numpy/torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    def _torch_seed(torch) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    register_post_import_hook(_torch_seed, 'torch')
