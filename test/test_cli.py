from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from glow.cli import parse_args


@dataclass
class Positional:
    arg: str


@dataclass
class UntypedList:  # Forbidden, as list field should always be typed
    args: list


@dataclass
class List_:  # noqa: N801
    args: list[str]


@dataclass
class UnsupportedTuple:
    args: tuple[str, int]


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
    param: str | None = None


@dataclass
class Optional_:  # noqa: N801
    param: str = 'hello'


@dataclass
class Nested:
    arg: str
    nested: Optional_


@dataclass
class NestedPositional:  # Forbidden, as only top level args can be positional
    arg2: str
    nested: Positional


@dataclass
class Aliased:
    arg: str = 'hello'


@dataclass
class NestedAliased:  # Forbidden as all field names must be unique
    arg: str
    nested: Aliased


@dataclass
class Custom:
    arg: Path


@pytest.mark.parametrize(
    ('argv', 'expected'),
    [
        (['value'], Positional('value')),
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
        (['test.txt'], Custom(Path('test.txt'))),
    ],
)
def test_good_class(argv: list[str], expected: Any):
    cls = type(expected)
    result, _ = parse_args(cls, argv)
    assert isinstance(result, cls)
    assert result == expected


@pytest.mark.parametrize(
    ('cls', 'exc_type'),
    [
        (Positional, SystemExit),
        (BadBoolean, ValueError),
        (UnsupportedTuple, TypeError),
        (UnsupportedSet, TypeError),
        (UntypedList, TypeError),
        (Nested, SystemExit),
        (NestedPositional, ValueError),
        (NestedAliased, ValueError),
    ],
)
def test_bad_class(cls: type[Any], exc_type: type[BaseException]):
    with pytest.raises(exc_type):
        parse_args(cls, [])


def _no_op():
    return ()


def _positional(a: int):
    return a


def _keyword(a: int = 4):
    return a


def _kw_nullable(a: int = None):  # type: ignore[assignment]  # noqa: RUF013
    return a


def _kwarg_literal(a: Literal[1, 2] = 1):
    return a


def _kwarg_bool(a: bool = False):
    return a


def _kwarg_list(a: list[int] = []):  # noqa: B006
    return a


def _kwarg_opt_list(a: list[int] | None = None):
    return a


def _arg_kwarg(a: int, b: str = 'hello'):
    return a, b


@pytest.mark.parametrize(
    ('argv', 'func', 'expected'),
    [
        ([], _no_op, ()),
        (['42'], _positional, 42),
        ([], _keyword, 4),
        (['--a', '58'], _keyword, 58),
        ([], _kw_nullable, None),
        (['--a', '73'], _kw_nullable, 73),
        ([], _kwarg_literal, 1),
        (['--a', '2'], _kwarg_literal, 2),
        ([], _kwarg_bool, False),
        (['--no-a'], _kwarg_bool, False),
        (['--a'], _kwarg_bool, True),
        ([], _kwarg_list, []),
        (['--a'], _kwarg_list, []),
        (['--a', '1'], _kwarg_list, [1]),
        (['--a', '1', '2'], _kwarg_list, [1, 2]),
        ([], _kwarg_opt_list, None),
        (['--a'], _kwarg_opt_list, []),
        (['--a', '1'], _kwarg_opt_list, [1]),
        (['--a', '1', '2'], _kwarg_opt_list, [1, 2]),
        (['53'], _arg_kwarg, (53, 'hello')),
        (['87', '--b', 'bye'], _arg_kwarg, (87, 'bye')),
    ],
)
def test_good_func[T](argv: list[str], func: Callable[..., T], expected: T):
    result, _ = parse_args(func, argv)
    assert type(result) is type(expected)
    assert result == expected
