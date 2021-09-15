from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

import pytest

from glow.cli import parse_args


@dataclass
class Arg:
    arg: str


@dataclass
class List_:
    args: list[str]


@dataclass
class UntypedList:  # Forbidden, as list field should always be typed
    args: list


@dataclass
class UnsupportedSet:
    args: set[str]


@dataclass
class BadBoolean:  # Forbidden, as boolean field should always have default
    arg: bool


@dataclass
class Boolean:
    param: bool = False


@dataclass
class Nullable:
    param: Optional[str] = None


@dataclass
class Optional_:
    param: str = 'hello'


@dataclass
class Nested:
    arg: str
    nested: Optional_


@dataclass
class NestedArg:  # Forbidden, as only top level args can be positional
    arg2: str
    nested: Arg


@dataclass
class Aliased:
    arg: str = 'hello'


@dataclass
class NestedAliased:  # Forbidden as all field names must be unique
    arg: str
    nested: Aliased


@pytest.mark.parametrize(('argv', 'result'), [
    (['value'], Arg('value')),
    ([], List_([])),
    (['a'], List_(['a'])),
    (['a', 'b'], List_(['a', 'b'])),
    ([], Boolean()),
    (['--no-param'], Boolean()),
    (['--param'], Boolean(True)),
    ([], Nullable()),
    (['--param', 'value'], Nullable('value')),
    ([], Optional_()),
    (['--param', 'world'], Optional_('world')),
    (['value'], Nested('value', Optional_())),
    (['value', '--param', 'pvalue'], Nested('value', Optional_('pvalue'))),
])
def test_cli_ok(argv: list[str], result: Any):
    cls = type(result)
    with patch('sys.argv', [''] + argv):
        obj, _ = parse_args(cls)
        assert isinstance(obj, cls)
        assert obj == result


@pytest.mark.parametrize(('cls', 'exc_type'), [
    (Arg, SystemExit),
    (BadBoolean, ValueError),
    (UnsupportedSet, ValueError),
    (UntypedList, ValueError),
    (Nested, SystemExit),
    (NestedArg, ValueError),
    (NestedAliased, ValueError),
])
def test_cli_fail(cls: type[Any], exc_type: type[BaseException]):
    with patch('sys.argv', ['']), pytest.raises(exc_type):
        parse_args(cls)
