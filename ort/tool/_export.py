import sys


class ExportError(Exception):
    pass


def export(obj):
    module = sys.modules[obj.__module__]
    if obj.__module__ == module.__spec__.parent:
        return obj

    name = obj.__name__
    namespace = sys.modules[module.__spec__.parent].__dict__

    __all__ = namespace.setdefault('__all__', [])
    if name in __all__ or name in namespace:
        raise ExportError(
            f'Name "{name}" is reserved in <{module.__spec__.parent}>'
            f' by <{namespace[name].__module__}:{name}>')

    namespace[name] = obj
    __all__.append(name)
    return obj


export = export(export)
