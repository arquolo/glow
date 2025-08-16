__all__ = ['Stream', 'cumsum', 'maximum_cumsum']

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True, slots=True, repr=False)
class Stream[Y, S]:
    init: S
    push: Callable[[S], None]
    pop: Callable[[], Y]

    def send(self, value: S) -> Y:
        self.push(value)
        return self.pop()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.send(self.init)})'


def cumsum() -> Stream[int, int]:
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
    todo = deque[int]()

    values = iter(todo.popleft, None)
    partial_sums = accumulate(values)

    return Stream(init=0, push=todo.append, pop=partial_sums.__next__)


def maximum_cumsum() -> Stream[int, int]:
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
    todo = deque[int]()

    values = iter(todo.popleft, None)
    partial_sums = accumulate(values)
    max_partial_sums = accumulate(partial_sums, max)

    return Stream(init=0, push=todo.append, pop=max_partial_sums.__next__)
