__all__ = ('export', 'import_tree')

import pkgutil
import sys


class ExportError(Exception):
    pass


def export(obj):
    """
    Exposes `obj` to `__all__` in module, parent to where it was defined.
    However it breaks intellisense.
    Example usage:

    ### `package/subpackage.py`:
    ```python
    from glow.api import export

    @export
    def func():
        pass
    ```
    ### `package/__init__.py`:
    ```python
    from . import subpackage
    ```
    ### `__main__.py`:
    ```python
    # not `from package.subpackage import func`
    from package import func
    func()
    ```
    """
    module = sys.modules[obj.__module__]
    if obj.__module__ == module.__spec__.parent:
        return obj

    name = obj.__name__
    namespace = sys.modules[module.__spec__.parent].__dict__

    __all__ = namespace.setdefault('__all__', [])
    if name in __all__ or name in namespace:
        raise ExportError(
            f'Name "{name}" is reserved in <{module.__spec__.parent}>' +
            f' by <{namespace[name].__module__}:{name}>',
        )

    namespace[name] = obj
    __all__.append(name)
    return obj


def import_tree(pkg: str):
    """
    Imports all subpackages. Example usage:

    ```python
    __import__('glow.api', fromlist=['api']).import_tree(__name__)
    ```
    """
    for _, name, __ in pkgutil.walk_packages(sys.modules[pkg].__path__):
        subpkg = pkg + '.' + name
        __import__(subpkg)
