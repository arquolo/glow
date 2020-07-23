import time
from dataclasses import dataclass
from itertools import count

import glow


@dataclass
class Value:
    value: int


def test_base():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=.05)
    assert ref._lock._loop is ref._loop


def test_fail():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=.05)
    assert ref.get().value == 0

    time.sleep(.10)
    assert ref.get().value == 1


def test_success() -> None:
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=.10)
    assert ref.get().value == 0

    time.sleep(.05)
    assert ref.get().value == 0


def test_success_double():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=.10)
    assert ref.get().value == 0

    time.sleep(.05)
    assert ref.get().value == 0

    time.sleep(.05)
    assert ref.get().value == 0
