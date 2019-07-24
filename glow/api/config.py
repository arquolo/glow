__all__ = 'Default', 'patch',

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from unittest import mock

_DEFAULT = object()


@dataclass
class Default:
    value: object = _DEFAULT

    def get_or(self, value):
        if self.value is _DEFAULT:
            return value
        return self.value


@contextmanager
def patch(obj, **kwargs):
    with ExitStack() as stack:
        for key, value in kwargs.items():
            handle = getattr(obj, key)
            stack.enter_context(
                mock.patch.object(handle, 'value', value)
                if isinstance(handle, Default)
                else mock.patch.object(obj, key, value)
            )
        yield