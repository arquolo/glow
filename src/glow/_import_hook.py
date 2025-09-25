__all__ = ['register_post_import_hook', 'when_imported']

import sys
from collections.abc import Callable, Sequence
from importlib import abc, util
from importlib.machinery import ModuleSpec
from threading import RLock
from types import ModuleType

from ._types import Callback

_INITIALIZED = False
_LOCK = RLock()
_HOOKS: dict[str, list[Callback[ModuleType]]] = {}


class _ImportHookChainedLoader(abc.Loader):
    def __init__(self, loader: abc.Loader) -> None:
        self.loader = loader

    def _set_loader(self, module: ModuleType) -> None:
        undefined = object()
        if getattr(module, '__loader__', undefined) in (None, self):
            try:
                module.__loader__ = self.loader
            except AttributeError:
                pass

        if (spec := getattr(module, '__spec__', None)) is not None and getattr(
            spec, 'loader', None
        ) is self:
            spec.loader = self.loader

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._set_loader(module)
        self.loader.exec_module(module)

        name: str | None = getattr(module, '__name__', None)
        with _LOCK:
            hooks = _HOOKS.pop(name, [])  # type: ignore[arg-type]
        for hook in hooks:
            hook(module)


class _ImportHookFinder(abc.MetaPathFinder, set[str]):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
        /,
    ) -> ModuleSpec | None:
        with _LOCK:
            if fullname not in _HOOKS or fullname in self:
                return None

        self.add(fullname)
        try:
            if (
                (spec := util.find_spec(fullname))
                and (loader := spec.loader)
                and not isinstance(loader, _ImportHookChainedLoader)
            ):
                spec.loader = _ImportHookChainedLoader(loader)
                return spec
        finally:
            self.remove(fullname)
        return None


def register_post_import_hook(hook: Callback[ModuleType], name: str) -> None:
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

        if (module := sys.modules.get(name)) is None:
            _HOOKS.setdefault(name, []).append(hook)
        else:
            hook(module)


def when_imported[H: Callback[ModuleType]](name: str) -> Callable[[H], H]:
    """Create decorator making a function a post import hook for a module.

    Simplified version of wrapt.when_imported.
    """

    def wrapper(hook: H) -> H:
        register_post_import_hook(hook, name)
        return hook

    return wrapper
