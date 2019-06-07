__all__ = ('patch', )

from contextlib import contextmanager, ExitStack
from unittest import mock


@contextmanager
def patch(obj, **kwargs):
    with ExitStack() as stack:
        for key, value in kwargs.items():
            stack.enter_context(mock.patch.object(obj, key, value))
        yield


# import functools
# from inspect import Signature, Parameter


# class Mock:
#     def __init__(self, x=None):
#         if x is None:
#             return
#         if isinstance(x, type(self)):
#             x = vars(x)
#         self.__dict__.update(
#             (k, (type(self)(v) if isinstance(v, dict) else v))
#             for k, v in x.items()
#         )

#     def __dir__(self):
#         return sorted(self.__dict__)

#     def __iter__(self):  # for conversion to dict
#         for k in dir(self):
#             v = getattr(self, k)
#             yield k, (dict(v) if isinstance(v, type(self)) else v)

#     def __repr__(self):
#         return f'{type(self).__name__}({vars(self)!r})'

#     def __eq__(self, other):
#         return type(other) == type(self) and vars(self) == vars(other)

#     def get(self, path) -> 'Mock':
#         factory = type(self)
#         return functools.reduce(
#             (lambda obj, name: vars(obj).setdefault(name, factory())),
#             path.split('.'), self
#         )

#     def update(self, path, **kwargs):
#         vars(self.get(path)).update(kwargs)

#     @property
#     def __call__(self):
#         @contextlib.contextmanager
#         def context(**kwargs):
#             state = vars(self).copy()
#             vars(self).update(kwargs)
#             try:
#                 yield
#             finally:
#                 vars(self).clear()
#                 vars(self).update(state)

#         parameters = (
#             Parameter(k, Parameter.POSITIONAL_OR_KEYWORD, default=v)
#             for k, v in vars(self).items() if not isinstance(v, type(self))
#         )
#         context.__signature__ = Signature(parameters=parameters)
#         return context


# def capture(fn=None, prefix: str = None, root: Mock = 'ROOT'):
#     if fn is None:
#         return functools.partial(capture, prefix=prefix, root=root)

#     sig = Signature.from_callable(fn)
#     params = [p for p in sig.parameters.values() if p.default is not p.empty]
#     if any(p.kind == p.KEYWORD_ONLY for p in params):
#         params = (p for p in params if p.kind == p.KEYWORD_ONLY)
#     defaults = {p.name: p.default for p in params}

#     if isinstance(root, str) and not isinstance(root, Mock):
#         root = globals()[root]
#     root.update(prefix, **defaults)

#     def wrapper(*args, **kwargs):
#         overrides = dict(root.get(prefix))
#         extra = set(overrides) - set(defaults)
#         if extra:
#             raise SyntaxError(
#                 f'Function <{fn.__module__}.{fn.__name__}{sig}>'
#                 f' cannot be called with {extra} parameters'
#             )

#         for k, v in defaults.items():
#             if overrides[k] == v:
#                 del overrides[k]
#         overrides.update(kwargs)
#         return fn(*args, **overrides)  # default < overrides < custom

#     return wrapper


# ROOT = Mock()
