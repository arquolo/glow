__all__ = ['cumsum', 'maximum_cumsum']

from collections import deque
from itertools import accumulate

from ._types import Callback, Get, Pipe


class _Pipe[In, Out](Pipe):
    def __init__(self, zero: In, push: Callback[In], pop: Get[Out]) -> None:
        self._zero = zero
        self._push = push
        self._pop = pop

    def send(self, value: In) -> Out:
        self._push(value)
        return self._pop()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.send(self._zero)})'


def cumsum() -> Pipe[int, int]:
    """Stream running cumulative sum.

    Coroutine version of:
        >>> numbers = [-1, -2, 3, -4, 5, 7]
        ... np.cumsum(numbers)
        [-1, -3, 0, -4, 1, 8]

    Usage:
        >>> m = cumsum()
        ... numbers = [-1, -2, 3, -4, 5, 7]
        ... [m.send(x) for x in numbers]
        [-1, -3, 0, -4, 1, 8]
    """
    buf = deque[int]()
    return _Pipe(
        zero=0,
        push=buf.append,
        pop=accumulate(iter(buf.popleft, None)).__next__,
    )


def maximum_cumsum() -> Pipe[int, int]:
    """Stream running maximum cumulative sum.

    Coroutine version of:
        >>> numbers = [1, -1, 1, 1, -1, -1]
        ... np.maximum.accumulate(np.cumsum(numbers))
        [1, 1, 1, 2, 2, 2]

    Usage:
        >>> m = maximum_cumsum()
        ... numbers = [1, -1, 1, 1, -1, -1]
        ... [m.send(x) for x in numbers]
        [1, 1, 1, 2, 2, 2]
    """
    buf = deque[int]()

    values = iter(buf.popleft, None)
    partial_sums = accumulate(values)
    max_partial_sums = accumulate(partial_sums, max)

    return _Pipe(zero=0, push=buf.append, pop=max_partial_sums.__next__)
