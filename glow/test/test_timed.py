import time
from enum import Enum

import pytest
from glow.decos import Timed


class Status(Enum):
    SUCCESS = 'success'
    FAIL = 'fail'


def test_fail():
    value = Timed(Status.FAIL, timeout=.03)
    time.sleep(.06)
    with pytest.raises(TimeoutError):
        value.get()


def test_success():
    value = Timed(Status.SUCCESS, timeout=.06)
    time.sleep(.03)
    assert value.get() == Status.SUCCESS


def test_success_double():
    value = Timed(Status.SUCCESS, timeout=.06)
    time.sleep(.03)
    value.get()
    time.sleep(.03)
    assert value.get() == Status.SUCCESS
