import pytest

from glow.api.config import ROOT, capture


@capture(prefix='test')
def function(param='default'):
    return param


def test_default():
    assert function() == 'default'


def test_custom():
    assert function(param='custom') == 'custom'


def test_override_default():
    with ROOT.test(param='override'):
        assert function() == 'override'


def test_never_override_custom():
    with ROOT.test(param='override'):
        assert function(param='custom') == 'custom'


def test_react_to_wrong_keyword():
    with pytest.raises(SyntaxError):
        with ROOT.test(value=10):
            function()
