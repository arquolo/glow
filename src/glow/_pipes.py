__all__ = ['Actor', 'cumsum', 'maximum_cumsum']

from collections import deque
from collections.abc import Iterator
from itertools import accumulate

from ._types import Pipe, Unary


class Actor[In, Out](Pipe[In, Out]):
    """Wrap an iterator transformation.

    Each `send(item)` queues an input for `fn` and returns the next item
    from its output iterator. `fn` is called once with an iterator over
    queued inputs. If its output iterator is exhausted, `send` raises
    `StopIteration`.

    >>> from itertools import accumulate
    >>> actor = Actor(accumulate)
    >>> actor.send(1)
    1
    >>> actor.send(2)
    3
    >>> actor.send(3)
    6

    """

    def __init__(self, fn: Unary[Iterator[In], Iterator[Out]], /) -> None:
        self._buf = deque[In]()
        # If `In` never None, we could use stackless source
        # instead of generator from `unqueue`
        self._iter = fn(iter(self._buf.popleft, object()))

    def send(self, value: In, /) -> Out:
        self._buf.append(value)
        return self._iter.__next__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._iter!r})'


def cumsum() -> Actor[int, int]:
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
    return Actor(accumulate)


def maximum_cumsum() -> Actor[int, int]:
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
    return Actor(lambda values: accumulate(accumulate(values), max))
