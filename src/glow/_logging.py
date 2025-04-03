__all__ = ['init_loguru']

import logging
import sys
from collections.abc import Iterable
from types import FrameType, ModuleType
from typing import Any

from loguru import logger

_DEFAULT_FMT = (
    '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>'
    ' | '
    '<level>{level: <8}</level>'
    ' | '
    '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>'
    ' | '
    '<level>{message}</level>'
)


def init_loguru(
    level: str = 'WARNING',
    *,
    names: Iterable[str] | dict[str, list[str]] = (),
    fmt: str = _DEFAULT_FMT,
    serialize: bool = False,
) -> None:
    """
    Configure loguru to:

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

    logger.remove()
    logger.add(sys.stdout, level=level, format=fmt, serialize=serialize)
    _intercept_std_logger('', level)

    if not isinstance(names, dict):
        names = {level: [*names]}

    names[level] = [
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
        *names[level],
    ]
    for level_, names_ in names.items():
        for name in names_:
            _intercept_std_logger(name, level_)


def _intercept_std_logger(module: str | ModuleType, level: int | str) -> None:
    log = logging.getLogger(
        module if isinstance(module, str) else module.__name__
    )
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
        depth = 1
        while frame and 'logging' in frame.f_code.co_filename:
            frame = frame.f_back
            depth += 1

        opt = logger.opt(exception=record.exc_info, depth=depth)
        opt.log(level, record.getMessage())
