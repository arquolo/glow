import pytest

from glow import api


class Factory:
    param = 'default'

    @classmethod
    def new(cls, param=None):
        if param is None:
            return cls.param
        return param


class FactoryPH:
    param = api.Default()

    @classmethod
    def new(cls, param=None):
        if param is None:
            return cls.param.get_or('default')
        return param


@pytest.mark.parametrize('test_cls', [Factory, FactoryPH])
@pytest.mark.parametrize('param,expected', [
    (None, 'default'),
    ('custom', 'custom'),
])
def test_default(test_cls, param, expected):
    assert test_cls.new(param=param) == expected


@pytest.mark.parametrize('test_cls', [Factory, FactoryPH])
@pytest.mark.parametrize('param,expected', [
    (None, 'override'),
    ('custom', 'custom'),
])
def test_override_default(test_cls, param, expected):
    with api.patch(test_cls, param='override'):
        assert test_cls.new(param=param) == expected


# def test_react_to_wrong_keyword():
#     with pytest.raises(SyntaxError):
#         with ROOT.test(value=10):
#             function()
