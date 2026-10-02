import numpy as np
import pytest

from glow import aminmax_norm


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32'])
@pytest.mark.parametrize('near_max', [False, True])
def test_normalization_full_range(dtype, near_max):
    maximum = int(np.iinfo(dtype).max)
    lo = maximum - 4 if near_max else 1
    source = np.array([[lo, lo + 1], [lo + 2, lo + 4]], dtype=dtype)
    original = source.copy()
    expected = np.array(
        [[0, (maximum + 2) // 4], [(maximum + 1) // 2, maximum]],
        dtype=dtype,
    )
    with np.errstate(all='raise'):
        actual = aminmax_norm(source)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == source.dtype


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32'])
def test_normalization_constant_and_full_range(dtype):
    for source in (
        np.array([7, 7], dtype=dtype),
        np.array([0, np.iinfo(dtype).max], dtype=dtype),
    ):
        np.testing.assert_array_equal(aminmax_norm(source), source)
