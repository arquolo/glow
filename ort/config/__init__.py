import contextlib
import functools
import unittest
from inspect import Signature, Parameter


class Attr:
    def __init__(self, x=None):
        if x is None:
            return
        if isinstance(x, type(self)):
            x = vars(x)
        self.__dict__.update(
            (k, (type(self)(v) if isinstance(v, dict) else v))
            for k, v in x.items()
        )

    def __setattr__(self, k, v):
        if isinstance(v, dict) and not isinstance(v, type(self)):
            v = type(self)(v)
        object.__setattr__(self, k, v)

    def __dir__(self):
        return sorted(self.__dict__)

    def __iter__(self):  # for conversion to dict
        for k in dir(self):
            v = getattr(self, k)
            yield k, (dict(v) if isinstance(v, type(self)) else v)

    def __repr__(self):
        return f'{type(self).__name__}({vars(self)!r})'

    def __eq__(self, other):
        return type(other) == type(self) and vars(self) == vars(other)

    def get(self, path):
        factory = type(self)
        return functools.reduce(
            (lambda obj, name: vars(obj).setdefault(name, factory())),
            path.split('.'), self
        )

    def update(self, path, **kwargs):
        vars(self.get(path)).update(kwargs)


class Descriptor:
    def __get__(self, obj, obj_type=None):
        @contextlib.contextmanager
        def context(**kwargs):
            state = vars(obj).copy()
            vars(obj).update(kwargs)
            try:
                yield
            finally:
                vars(obj).clear()
                vars(obj).update(state)

        parameters = (
            Parameter(k, Parameter.POSITIONAL_OR_KEYWORD, default=v)
            for k, v in vars(obj).items() if not isinstance(v, obj_type)
        )
        context.__signature__ = Signature(parameters=parameters)
        return context


class Feature(Attr):
    __call__ = Descriptor()


def capture(fn=None, prefix=None, root='ROOT'):
    if fn is None:
        return functools.partial(capture, prefix=prefix)

    sig = Signature.from_callable(fn)
    params = [p for p in sig.parameters.values() if p.default is not p.empty]
    if any(p.kind == p.KEYWORD_ONLY for p in params):
        params = (p for p in params if p.kind == p.KEYWORD_ONLY)
    defaults = {p.name: p.default for p in params}

    if isinstance(root, str) and not isinstance(root, Feature):
        root = globals()[root]
    root.update(prefix, **defaults)

    def wrapper(*args, **kwargs):
        overrides = dict(root.get(prefix))
        extra = set(overrides) - set(defaults)
        if extra:
            raise SyntaxError(
                f'Function <{fn.__module__}.{fn.__name__}{sig}>'
                f' cannot be called with {extra} parameters'
            )

        for k, v in defaults.items():
            if overrides[k] == v:
                del overrides[k]
        overrides.update(kwargs)
        return fn(*args, **overrides)  # default < overrides < custom

    return wrapper


ROOT = Feature()


class _TestSuite(unittest.TestCase):
    def setUp(self):
        @capture(prefix='test')
        def test(param='default'):
            return param

        self.test_function = test

    def test_default(self):
        self.assertEqual(self.test_function(), 'default')

    def test_custom(self):
        self.assertEqual(self.test_function(param='custom'), 'custom')

    def test_default_override(self):
        """Should override default calls"""
        with ROOT.test(param='default_override'):
            self.assertEqual(self.test_function(), 'default_override')

    def test_custom_override(self):
        """Shouldn't override custom calls"""
        with ROOT.test(param='custom_override'):
            self.assertEqual(self.test_function(param='custom'), 'custom')

    def test_wrong_keyword(self):
        with self.assertRaises(SyntaxError):
            with ROOT.test(value=10):
                self.test_function()
