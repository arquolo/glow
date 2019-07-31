from glow.api import Default, patch
# import pytest


class Factory:
    param = 'default'

    @classmethod
    def new(cls, param=None):
        if param is None:
            return cls.param
        return param


class FactoryPH:
    param = Default()

    @classmethod
    def new(cls, param=None):
        if param is None:
            return cls.param.get_or('default')
        return param


def test_default():
    assert Factory.new() == 'default'


def test_default_ph():
    assert FactoryPH.new() == 'default'


def test_custom():
    assert Factory.new(param='custom') == 'custom'


def test_custom_ph():
    assert FactoryPH.new(param='custom') == 'custom'


def test_override_default():
    with patch(Factory, param='override'):
        assert Factory.new() == 'override'


def test_override_default_ph():
    with patch(FactoryPH, param='override'):
        assert FactoryPH.new() == 'override'


def test_never_override_custom():
    with patch(Factory, param='override'):
        assert Factory.new(param='custom') == 'custom'


def test_never_override_custom_ph():
    with patch(FactoryPH, param='override'):
        assert FactoryPH.new(param='custom') == 'custom'


# def test_react_to_wrong_keyword():
#     with pytest.raises(SyntaxError):
#         with ROOT.test(value=10):
#             function()
