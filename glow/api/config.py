__all__ = ['patch', 'Default']

import contextlib
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


@contextlib.contextmanager
def patch(obj, **kwargs):
    with contextlib.ExitStack() as stack:
        for key, value in kwargs.items():
            proxy = getattr(obj, key)
            args = ((proxy, 'value') if isinstance(proxy, Default) else
                    (obj, key))
            stack.enter_context(mock.patch.object(*args, value))
        yield
