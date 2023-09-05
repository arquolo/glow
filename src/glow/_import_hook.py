__all__ = ['register_post_import_hook', 'when_imported']

import sys
from collections.abc import Callable
from importlib import abc, util
from threading import RLock
from typing import Any, TypeVar

_Hook = Callable[[Any], object]
_HookVar = TypeVar('_HookVar', bound=_Hook)

_INITIALIZED = False
_LOCK = RLock()
_HOOKS: dict[str, list[_Hook]] = {}


class _ImportHookChainedLoader(abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def _set_loader(self, module):
        undefined = object()
        if getattr(module, '__loader__', undefined) in (None, self):
            try:
                module.__loader__ = self.loader
            except AttributeError:
                pass

        if ((spec := getattr(module, '__spec__', None)) is not None
                and getattr(spec, 'loader', None) is self):
            spec.loader = self.loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self._set_loader(module)
        self.loader.exec_module(module)

        name: str | None = getattr(module, '__name__', None)
        with _LOCK:
            hooks = _HOOKS.pop(name, [])  # type: ignore[arg-type]
        for hook in hooks:
            hook(module)


class _ImportHookFinder(abc.MetaPathFinder, set[str]):
    def find_spec(self, fullname: str, path, target=None):
        with _LOCK:
            if fullname not in _HOOKS or fullname in self:
                return None

        self.add(fullname)
        try:
            if ((spec := util.find_spec(fullname)) and (loader := spec.loader)
                    and not isinstance(loader, _ImportHookChainedLoader)):
                spec.loader = _ImportHookChainedLoader(loader)
                return spec
        finally:
            self.remove(fullname)
        return None


def register_post_import_hook(hook: _Hook, name: str) -> None:
    """Register a new post import hook for the target module name.

    This will result in a proxy callback being registered which will defer
    loading of the specified module containing the callback function until
    required.

    Simplified version of wrapt.register_post_import_hook.
    """
    with _LOCK:
        global _INITIALIZED  # noqa: PLW0603
        if not _INITIALIZED:
            _INITIALIZED = True
            sys.meta_path.insert(0, _ImportHookFinder())

        if (module := sys.modules.get(name)) is not None:
            hook(module)
        else:
            _HOOKS.setdefault(name, []).append(hook)


def when_imported(name: str) -> Callable[[_HookVar], _HookVar]:
    """
    Decorator for marking that a function should be called as a post
    import hook when the target module is imported.

    Simplified version of wrapt.when_imported.
    """
    def wrapper(hook: _HookVar) -> _HookVar:
        register_post_import_hook(hook, name)
        return hook

    return wrapper
