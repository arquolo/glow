__all__ = ['patch', 'Default']

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any
from unittest import mock

_DEFAULT = object()
# TODO: just use collections.ChainMap


@dataclass
class Default:
    value: Any = _DEFAULT

    def get_or(self, value):
        if self.value is _DEFAULT:
            return value
        return self.value


@contextmanager
def patch(obj, **kwargs):
    with ExitStack() as stack:
        for key, value in kwargs.items():
            proxy = getattr(obj, key)
            if isinstance(proxy, Default):
                stack.enter_context(mock.patch.object(proxy, 'value', value))
            else:
                stack.enter_context(mock.patch.object(obj, key, value))
        yield
