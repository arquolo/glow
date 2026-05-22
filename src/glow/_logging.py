__all__ = ['init_loguru', 'span_task']

import logging
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import FrameType
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

from loguru import logger

if TYPE_CHECKING:
    from loguru import FilterDict, FilterFunction

_DEFAULT_FMT = (
    '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>'
    ' | '
    '<level>{level: <8}</level>'
    ' | '
    '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>'
    ' | '
    '<level>{message}</level>'
)
_DEFAULT_MODULES = [
    'uvicorn',
    'uvicorn.access',
    'uvicorn.error',
    'hypercorn',
    'hypercorn.access',
    'hypercorn.error',
    'sse_starlette.sse',
    'websockets',
    'websockets.client',
    'websockets.protocol',
    'websockets.server',
]


class _LoggerAddKwds(TypedDict, total=False):
    colorize: bool | None
    serialize: bool
    backtrace: bool
    diagnose: bool
    filter: 'str | FilterFunction | FilterDict'


def init_loguru(
    level: str = 'WARNING',
    *,
    names: Iterable[str] | Mapping[str, Sequence[str]] = (),
    fmt: str = _DEFAULT_FMT,
    extra: bool = False,
    **logger_add_kwargs: Unpack[_LoggerAddKwds],
) -> None:
    """
    Configure loguru.

    Does:
    - remap all `logging.Logger` to `loguru` calls
    - remap all warnings to `logger.warning`
    - configure `uvicorn`, `hypercorn` and `websockets` loggers.
    """
    logging.basicConfig(
        level=level,
        handlers=[_InterceptHandler()],
        force=True,
    )
    logging.captureWarnings(True)

    if extra:
        fmt = fmt + ' | {extra}'
    logger.remove()
    logger.add(sys.stdout, level=level, format=fmt, **logger_add_kwargs)
    _intercept_std_logger('', level)

    for modname in _DEFAULT_MODULES:
        _intercept_std_logger(modname, level)

    if isinstance(names, Mapping):
        for level_, names_ in names.items():
            for modname in names_:
                _intercept_std_logger(modname, level_)
    else:
        for modname in names:
            _intercept_std_logger(modname, level)


def _intercept_std_logger(modname: str, level: int | str) -> None:
    log = logging.getLogger(modname)
    log.handlers = [_InterceptHandler(level=level)]
    log.propagate = False


class _InterceptHandler(logging.Handler):
    def emit(self, record: Any) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame: FrameType | None = logging.currentframe()
        depth = 0
        while frame and 'logging' in frame.f_code.co_filename:
            frame = frame.f_back
            depth += 1

        opt = logger.opt(exception=record.exc_info, depth=depth)
        opt.log(level, record.getMessage())


@contextmanager
def span_task(task_id: str) -> Iterator[str]:
    span = _span_ctx.get({})
    if parent_id := span.get('task_id'):
        task_id = f'{parent_id}/{task_id}'

    token = _span_ctx.set(span | {'task_id': task_id})
    try:
        with logger.contextualize(task_id=task_id):
            yield task_id
    finally:
        _span_ctx.reset(token)


_span_ctx = ContextVar[dict[str, str]]('span')
