import pytest


@pytest.fixture
def make_squares():
    import glow

    @glow.as_sized(hint=len)
    def _make_squares(it):
        return (x ** 2 for x in it)

    return _make_squares


def test_ok(make_squares):
    squares = make_squares(range(5))

    assert len(squares) == 5
    assert [*squares] == [0, 1, 4, 9, 16]
    assert len(squares) == 0


def test_ok_while(make_squares):
    squares = make_squares(range(5))

    assert len(squares) == 5
    while len(squares):
        next(squares)
    pytest.raises(StopIteration, next, squares)


def test_fail(make_squares):
    squares = make_squares(x for x in range(5))

    pytest.raises(TypeError, len, squares)
    assert [*squares] == [0, 1, 4, 9, 16]


def test_windowed():
    import glow
    it = glow.windowed(range(5), 3)

    assert len(it) == 3
    assert [*it] == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert len(it) == 0


def test_sliced():
    import glow
    it = glow.sliced(range(5), 3)

    assert len(it) == 2
    assert [*it] == [range(0, 3), range(3, 5)]
    assert len(it) == 0


def test_chunked():
    import glow
    it = glow.chunked(range(5), 3)

    assert len(it) == 2
    assert [*it] == [(0, 1, 2), (3, 4)]
    assert len(it) == 0
