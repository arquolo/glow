__all__ = ['trace', 'trace_module', 'whereami']

from collections.abc import Callable
from contextlib import suppress
from types import ModuleType

import wrapt

from ._import_hook import register_post_import_hook
from ._profile import whereami


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
