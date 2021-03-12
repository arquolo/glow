from __future__ import annotations  # until 3.10

__all__ = ['arg', 'parse_args']

import sys
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      BooleanOptionalAction)
from collections.abc import Sequence
from dataclasses import MISSING, field, is_dataclass
from typing import TypeVar, Union, get_args, get_origin, get_type_hints

_T = TypeVar('_T')


def arg(
        *,
        default=MISSING,
        default_factory=MISSING,
        init=True,
        repr=True,  # noqa: A002
        hash=None,  # noqa: A002
        help=None,  # noqa: A002
        compare=True,
        metadata=None):
    """Alias for dataclass.field with help"""
    metadata = metadata or {}
    if help:
        metadata = {**metadata, 'help': help}
    return field(  # type: ignore
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata)


def _parse(
        cls: type[_T],
        cmdline: Sequence[str]) -> tuple[ArgumentParser, _T, list[str]]:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    nested: dict = {}
    hints = get_type_hints(cls)
    for name, field_ in cls.__dataclass_fields__.items():  # type: ignore
        type_ = hints[name]  # Postponed annotation
        help_ = field_.metadata.get('help')
        default = field_.default

        snake = name.replace('_', '-')

        # nested
        # TODO: Add support for nested dataclasses
        # if is_dataclass(type_):
        #     _, inst, cmdline = _parse(type_, cmdline)
        #     nested[name] = inst

        # optional
        if type_ is bool:
            if default is MISSING:
                raise ValueError(f'Boolean field "{name}" must have default')
            parser.add_argument(
                f'--{snake}',
                action=BooleanOptionalAction,
                default=default,
                help=help_)
        elif default is not MISSING:
            origin = get_origin(type_)
            args = get_args(type_)
            # strip Optional
            if origin is Union and len(args) == 2 and type(None) in args:
                type_, = (a for a in args if not issubclass(a, type(None)))
            parser.add_argument(
                f'--{snake}', default=default, type=type_, help=help_)

        # positional with nargs
        elif (origin := get_origin(type_)) and (args := get_args(type_)):
            if origin is Union and len(args) == 2 and type(None) in args:
                type_, = (a for a in args if not issubclass(a, type(None)))
                parser.add_argument(snake, nargs='?', type=type_, help=help_)
            elif origin is list:
                type_, = args
                parser.add_argument(snake, nargs='+', type=type_, help=help_)
            else:
                raise ValueError(
                    'Only List[...] and Optional[...] are supported. '
                    f'Got: {type_}')

        # pure positional
        else:
            parser.add_argument(snake, type=type_, help=help_)

    namespace, cmdline = parser.parse_known_args(cmdline)
    # namespace = parser.parse_args()
    return parser, cls(**vars(namespace), **nested), cmdline  # type: ignore


def parse_args(cls):
    """Use dataclass as source for parser"""
    assert is_dataclass(cls)

    parser, instance, rest = _parse(cls, sys.argv[1:])
    assert not rest
    return parser, instance
