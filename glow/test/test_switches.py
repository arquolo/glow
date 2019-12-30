import glow


def test_switch():
    def do_increment(latency):
        nonlocal value
        with glow.interpreter_lock(latency):
            for _ in range(100_000):
                value = value + 1
                # <- can switch here ->
                value = value - 1

    values = []
    for latency in (1e3, 1e-6):
        value = 0
        glow.eat(glow.mapped(do_increment, (latency for _ in range(100))))
        values.append(abs(value))

    assert sorted(values) == values
    assert values[0] == 0
    assert values[-1] != 0
