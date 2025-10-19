"""argparse and dataclasses, married.

Example:
```
@dataclass
class Args:
    a: int
    b: str = 'hello'

args, parser = parse_args(Args)
```
Or with plain function:
```
@parse_args
def main(name: str = 'user'):
    print(f'Hello {name}')
```

Reasons not to use alternatives:
- [simple-parsing](https://github.com/lebrice/SimpleParsing):
  - Has underscores (`--like_this`) instead of dashes (`--like-this`)
  - Erases type on parser result, thus making typed prototype useless
    (what is the point of using dataclasses if it is passed
     through function returning typing.Any/argparse.Namespace?)

- [datargs](https://github.com/roee30/datargs):
  - No nesting support
  - No function's support

- [pydantic](https://github.com/samuelcolvin/pydantic):
  - supports CLI via BaseSettings and environment variables parsing
  - no nesting, as requires mixing only via multiple inheritance

- [typer](https://github.com/tiangolo/typer):
  - No support on dataclasses
    (https://github.com/tiangolo/typer/issues/154).
  - No fine way to extract parsed options without invoking because of
    decorator/callback based implementation. Thus enforces wrapping of the
    whole app into `typer.run`.
    (https://github.com/tiangolo/typer/issues/197).
"""

__all__ = ['arg', 'parse_args']

import argparse
import importlib
import sys
import types
from argparse import ArgumentParser, BooleanOptionalAction, _ArgumentGroup
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from inspect import getmodule, signature, stack
from typing import (
    Any,
    Literal,
    Required,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_inspection.introspection import (
    UNKNOWN,
    AnnotationSource,
    inspect_annotation,
)

type _Node = str | tuple[str, type, list['_Node']]


@dataclass(kw_only=True)
class Meta:
    help: str = ''
    flag: str | None = None


@dataclass(kw_only=True)
class _Meta(Meta):
    name: str


def arg(
    default=MISSING,
    /,
    *,
    flag=None,
    factory=MISSING,
    init=True,
    repr=True,  # noqa: A002
    hash=None,  # noqa: A002
    help=None,  # noqa: A002
    compare=True,
    metadata=None,
) -> Field:
    """Annotate dataclass.field with extra metadata (like help)."""
    metadata = metadata or {}
    for k, v in {'flag': flag, 'help': help}.items():
        if v:
            metadata = metadata | {k: v}
    return field(  # type: ignore[call-overload]
        default=default,
        default_factory=factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


class _Opts(TypedDict, total=False):
    type: Required[Callable]
    nargs: str
    choices: Iterable


def _unwrap_type(tp: type) -> tuple[type, _Opts]:
    origin = get_origin(tp)
    args = get_args(tp)
    if not origin or not args:  # Not a generic type
        return tp, {'type': tp}

    if origin is list:  # `List[T]`
        cls, opts = _unwrap_type(args[0])
        return cls, {**opts, 'nargs': argparse.ZERO_OR_MORE}

    if (  # `Optional[T]` or `T | None`
        origin in {Union, types.UnionType}
        and len(args) == 2
        and types.NoneType in args
    ):
        [tp_] = set(args) - {types.NoneType}
        cls, opts = _unwrap_type(tp_)
        if opts.get('nargs') == argparse.ZERO_OR_MORE:
            return cls, opts
        return cls, {**opts, 'nargs': argparse.OPTIONAL}

    if origin is Literal:  # `Literal[x, y]`
        choices = get_args(tp)
        if len(tps := {type(c) for c in choices}) != 1:
            msg = f'Literal parameters should have the same type. Got: {tps}'
            raise TypeError(msg)
        [cls] = tps
        return cls, {'type': cls, 'choices': choices}

    msg = (
        'Only list, Optional and Literal are supported as generic types. '
        f'Got: {tp}'
    )
    raise TypeError(msg)


def _get_fields(fn: Callable) -> Iterator[Field]:
    if is_dataclass(fn):  # Shortcut
        yield from fields(fn)
        return

    for p in signature(fn).parameters.values():
        if p.kind is p.KEYWORD_ONLY and p.default is p.empty:
            msg = f'Keyword "{p.name}" must have default'
            raise ValueError(msg)
        if p.kind in {p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD}:
            msg = f'Unsupported parameter type: {p.kind}'
            raise TypeError(msg)

        if isinstance(p.default, Field):
            fd = p.default
        else:
            fd = arg(MISSING if p.default is p.empty else p.default)
        fd.name = p.name
        yield fd


def _get_metadata(tp: type, fd: Field) -> tuple[type, _Meta]:
    info = inspect_annotation(tp, annotation_source=AnnotationSource.CLASS)

    flag = fd.metadata.get('flag')
    name = fd.name.replace('_', '-')
    help_ = fd.metadata.get('help') or ''

    if info.type is not UNKNOWN:
        tp = info.type
        for m in info.metadata:
            if isinstance(m, Meta):
                help_ = m.help
                flag = m.flag

    return tp, _Meta(help=help_, flag=flag, name=name)


def _visit_nested(
    parser: ArgumentParser | _ArgumentGroup,
    fn: Callable,
    seen: dict[str, list],
) -> list[_Node]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except NameError:
        if fn.__module__ != '__main__':
            raise
        for finfo in stack():
            if not getmodule(f := finfo.frame):
                hints = get_type_hints(fn, f.f_globals, include_extras=True)
                break
        else:
            raise

    nodes: list[_Node] = []
    for fd in _get_fields(fn):
        if fd.init:
            seen.setdefault(fd.name, []).append(fn)
            nodes.append(_visit_field(parser, hints[fd.name], fd, seen))

    for name, usages in seen.items():
        if len(usages) > 1:
            msg = (
                f'Field name "{name}" occured multiple times: '
                + ', '.join(f'{c.__module__}.{c.__qualname__}' for c in usages)
                + '. All field names should be unique'
            )
            raise ValueError(msg)
    return nodes


def _visit_field(
    parser: ArgumentParser | _ArgumentGroup,
    tp: type,
    fd: Field,
    seen: dict[str, list],
) -> _Node:
    tp, meta = _get_metadata(tp, fd)
    cls, opts = _unwrap_type(tp)

    if cls is not bool and fd.default is not MISSING:
        meta.help += f' (default: {fd.default})'

    if is_dataclass(cls):  # Nested dataclass
        arg_group = parser.add_argument_group(fd.name)
        return fd.name, cls, _visit_nested(arg_group, cls, seen)

    vtp = opts['type']
    if (
        isinstance(vtp, type)
        and issubclass(vtp, Iterable)
        and not issubclass(vtp, str)
    ):
        msg = (
            'Iterable value types are supported only as generics. '
            f'Got: {vtp}'
        )
        raise TypeError(msg)

    flags = [meta.flag] if meta.flag else []
    default = (
        fd.default if fd.default_factory is MISSING else fd.default_factory()
    )

    if cls is bool:  # Optional
        if default is MISSING:
            msg = f'Boolean field "{fd.name}" should have default'
            raise ValueError(msg)
        parser.add_argument(
            f'--{meta.name}',
            *flags,
            action=BooleanOptionalAction,
            default=default,
            help=meta.help,
        )

    # Generic optional
    elif default is not MISSING:
        if opts.get('nargs') == argparse.OPTIONAL:
            del opts['nargs']
        parser.add_argument(
            f'--{meta.name}', *flags, **opts, default=default, help=meta.help
        )

    elif isinstance(parser, ArgumentParser):  # Allow only for root parser
        if meta.flag:
            msg = f'Positional-only field "{fd.name}" should not have flag'
            raise ValueError(msg)
        parser.add_argument(meta.name, **opts, help=meta.help)

    else:
        msg = (
            'Positional-only fields are forbidden for nested types. '
            f'Please set default value for "{fd.name}"'
        )
        raise ValueError(msg)

    return fd.name


def _construct[T](
    src: dict[str, Any], fn: Callable[..., T], args: Iterable[_Node]
) -> T:
    kwargs = {}
    for a in args:
        if isinstance(a, str):
            kwargs[a] = src.pop(a)
        else:
            kwargs[a[0]] = _construct(src, a[1], a[2])
    return fn(**kwargs)


def parse_args[T](
    fn: Callable[..., T],
    /,
    args: Sequence[str] | None = None,
    prog: str | None = None,
) -> tuple[T, ArgumentParser]:
    """Create parser from type hints of callable, parse args and do call."""
    # TODO: Rename to `run`
    if not callable(fn):
        raise TypeError(f'Expectet callable. Got: {type(fn).__qualname__}')

    parser = ArgumentParser(prog)
    nodes = _visit_nested(parser, fn, {})

    assert args is None or (
        isinstance(args, Sequence) and not isinstance(args, str)
    )

    namespace = parser.parse_args(args)
    obj = _construct(vars(namespace), fn, nodes)
    return obj, parser


def _import_from_string(qualname: str):
    modname, _, attrname = qualname.partition(':')
    if not modname or not attrname:
        msg = (
            f'Import string "{qualname}" must be '
            'in format "<module>:<attribute>".'
        )
        raise ImportError(msg)

    mod = importlib.import_module(modname)

    obj: Any = mod
    try:
        for a in attrname.split('.'):
            obj = getattr(obj, a)
    except AttributeError:
        msg = f'Attribute "{attrname}" not found in module "{modname}".'
        raise AttributeError(msg) from None
    return obj


if __name__ == '__main__':
    qualname, *argv = sys.argv
    obj = _import_from_string(qualname)
    parse_args(obj, argv)
