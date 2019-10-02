import time
from dataclasses import dataclass
from itertools import count

from glow import Reusable


@dataclass
class Value:
    value: int


def test_fail():
    counter = count()
    ref = Reusable(lambda: Value(next(counter)), timeout=.03)
    assert ref.get().value == 0

    time.sleep(.06)
    assert ref.get().value == 1


def test_success():
    counter = count()
    ref = Reusable(lambda: Value(next(counter)), timeout=.06)
    assert ref.get().value == 0

    time.sleep(.03)
    assert ref.get().value == 0


def test_success_double():
    counter = count()
    ref = Reusable(lambda: Value(next(counter)), timeout=.06)
    assert ref.get().value == 0

    time.sleep(.03)
    assert ref.get().value == 0

    time.sleep(.03)
    assert ref.get().value == 0
