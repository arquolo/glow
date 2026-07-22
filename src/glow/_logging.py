__all__ = ['init_loguru', 'span_task']

import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from functools import update_wrapper
from types import FrameType
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

from loguru import logger

from ._dev import hide_frame

if TYPE_CHECKING:
    from loguru import FilterDict, FilterFunction, FormatFunction

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
    fmt: 'FormatFunction | str' = _DEFAULT_FMT,
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

    if extra and not callable(fmt):
        fmt = fmt + ' | {extra}'
    logger.remove()
    logger.add(sys.stdout, level=level, format=fmt, **logger_add_kwargs)

    handler = _InterceptHandler(level)
    for modname in ['', *_DEFAULT_MODULES]:
        _set_handler(modname, handler)

    if isinstance(names, Mapping):
        for lv, ns in names.items():
            h = _InterceptHandler(lv) if lv != level else handler
            for modname in ns:
                _set_handler(modname, h)
    else:
        for modname in names:
            _set_handler(modname, handler)


def _set_handler(modname: str, handler: logging.Handler) -> None:
    log = logging.getLogger(modname)
    log.handlers = [handler]
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


def span_task[**P, R](
    id_or_func: Callable[P, R] | str | None = None, /
) -> 'Callable[P, R] | _TaskSpanner | str | None':
    if id_or_func is None:
        return _span_ctx.get({}).get('task_id')
    if callable(id_or_func):
        return _TaskSpanner(id_or_func.__qualname__)(id_or_func)
    return _TaskSpanner(id_or_func)


class _TaskSpanner:
    """Adds task_id to loguru.logger's extra

    Could be used as decorator for function or async function,
    or just as a context manager.
    """

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._ctx: AbstractContextManager[str] | None = None

    def __enter__(self) -> str:
        if self._ctx:
            raise RuntimeError('nesting context managers is not allowed')
        self._ctx = _span_task(self._task_id)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        ctx, self._ctx = self._ctx, None
        if ctx is None:
            raise RuntimeError('__enter__ was not called')
        return ctx.__exit__(exc_type, exc, tb)

    def __call__[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn):
            raise RuntimeError(
                f'Generator functions are not supported. Got {fn}'
            )

        if inspect.iscoroutinefunction(fn):

            async def awrapper(*args: P.args, **kwargs: P.kwargs):
                with hide_frame, _span_task(self._task_id):
                    return await fn(*args, **kwargs)

            wrapper = cast('Callable[P, R]', awrapper)
        else:

            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with hide_frame, _span_task(self._task_id):
                    return fn(*args, **kwargs)

        return update_wrapper(wrapper, fn)


@contextmanager
def _span_task(task_id: str) -> Iterator[str]:
    span = _span_ctx.get({})
    if parent_id := span.get('task_id'):
        task_id = f'{parent_id}/{task_id}'

    # TODO: opentelemetry integration
    token = _span_ctx.set(span | {'task_id': task_id})
    try:
        with logger.contextualize(task_id=task_id):
            yield task_id
    finally:
        _span_ctx.reset(token)


_span_ctx = ContextVar[dict[str, str]]('span')
