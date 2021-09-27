from __future__ import annotations

__all__ = ['arg', 'parse_args']

from argparse import ArgumentParser, BooleanOptionalAction, _ArgumentGroup
from collections.abc import Iterable, Sequence
from dataclasses import MISSING, Field, field, is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

_T = TypeVar('_T')
_Node = Union[str, tuple[str, type[Any], list['_Node']]]  # type: ignore


def arg(
        default=MISSING,
        /,
        *,
        factory=MISSING,
        init=True,
        repr=True,  # noqa: A002
        hash=None,  # noqa: A002
        help=None,  # noqa: A002
        compare=True,
        metadata=None):
    """Convinient alias for dataclass.field with extra metadata (like help)"""
    metadata = metadata or {}
    if help:
        metadata = {**metadata, 'help': help}
    return field(  # type: ignore
        default=default,
        default_factory=factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata)


def _unwrap_type(tp: type) -> tuple[type, str | None]:
    if tp is list:
        raise ValueError('Type list should be parametrized')

    origin = get_origin(tp)
    *args, = get_args(tp)
    if not origin or not args:
        return tp, None

    if origin is list:
        return args[0], '*'

    if origin is Union:
        args.remove(type(None))
        if len(args) == 1:
            return args[0], '?'

    raise ValueError('Only list and Optional are supported '
                     f'as generic types. Got: {tp}')


def _prepare_nested(parser: ArgumentParser | _ArgumentGroup, cls: type,
                    seen: dict[str, list[type]]) -> list[_Node]:
    # TODO: Allow callable or class (instead of just class) for `cls` argument
    hints = get_type_hints(cls)
    fields: dict[str, Field] = cls.__dataclass_fields__  # type: ignore

    nodes = [
        _prepare_field(parser, hints[name], name, fd, seen)
        for name, fd in fields.items() if fd.init
    ]
    for name in fields:
        seen.setdefault(name, []).append(cls)
    for name, usages in seen.items():
        if len(usages) > 1:
            raise ValueError(f'Field name "{name}" occured multiple times: ' +
                             ', '.join(f'{c.__module__}.{c.__qualname__}'
                                       for c in usages) +
                             '. All field names should be unique')
    return nodes


def _prepare_field(parser: ArgumentParser | _ArgumentGroup, cls: type,
                   name: str, field_: Field, seen: dict[str, list]) -> _Node:
    cls, nargs = _unwrap_type(cls)

    help_ = field_.metadata.get('help') or ''
    default = field_.default
    if cls is not bool and default is not MISSING:
        help_ += f' (default: {default})'

    if is_dataclass(cls):  # Nested dataclass
        arg_group = parser.add_argument_group(name)
        return name, cls, _prepare_nested(arg_group, cls, seen)

    snake = name.replace('_', '-')

    if cls is bool:  # Optional
        if default is MISSING:
            raise ValueError(f'Boolean field "{name}" must have default')
        parser.add_argument(
            f'--{snake}',
            action=BooleanOptionalAction,
            default=default,
            help=help_)

    elif default is not MISSING:  # Generic optional
        parser.add_argument(
            f'--{snake}', default=default, type=cls, help=help_)

    elif isinstance(parser, ArgumentParser):  # Allow only for root parser
        if nargs is not None:  # N positionals
            parser.add_argument(snake, nargs=nargs, type=cls, help=help_)

        else:  # Positional
            parser.add_argument(snake, type=cls, help=help_)

    else:
        raise ValueError('Positionals are not allowed for nested classes')

    return name


def _construct(src: dict[str, Any], cls: type[_T],
               args: Iterable[_Node]) -> _T:
    kwargs = {}
    for a in args:
        if isinstance(a, str):
            kwargs[a] = src.pop(a)
        else:
            kwargs[a[0]] = _construct(src, a[1], a[2])
    return cls(**kwargs)  # type: ignore


def parse_args(cls: type[_T],
               args: Sequence[str] | None = None) -> tuple[_T, ArgumentParser]:
    """Use dataclass as source for parser"""
    # TODO: Rename to `exec_cli`
    assert is_dataclass(cls)

    parser = ArgumentParser()
    nodes = _prepare_nested(parser, cls, {})

    namespace = parser.parse_args(args)
    obj = _construct(vars(namespace), cls, nodes)
    return obj, parser
