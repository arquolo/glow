import time

import pytest
from glow.decos import Timed


def test_fail():
    value = Timed('fail', timeout=.03)
    time.sleep(.06)
    with pytest.raises(TimeoutError):
        value.get()


def test_success():
    value = Timed('success', timeout=.06)
    time.sleep(.03)
    assert value.get() == 'success'


def test_success_double():
    value = Timed('success', timeout=.06)
    time.sleep(.03)
    value.get()
    time.sleep(.03)
    assert value.get() == 'success'
