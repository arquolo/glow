from typing import Callable, ContextManager, Sequence, TypeVar, overload

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_BatchF = TypeVar('_BatchF', bound=Callable[[Sequence], list])


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    ...


def interpreter_lock(timeout: float = ...) -> ContextManager[None]:
    ...


def call_once(fn: _F) -> _F:
    ...


def shared_call(fn: _F) -> _F:
    ...


@overload
def stream_batched(*,
                   batch_size: int,
                   latency: float = ...,
                   timeout: float = ...) -> Callable[[_BatchF], _BatchF]:
    ...


@overload
def stream_batched(func: _BatchF,
                   *,
                   batch_size: int,
                   latency: float = ...,
                   timeout: float = ...) -> _BatchF:
    ...
