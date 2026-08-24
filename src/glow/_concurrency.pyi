from collections.abc import Callable

from ._types import Get

def threadlocal[T, **P](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Get[T]: ...
