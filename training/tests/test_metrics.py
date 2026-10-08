from shellwise_train.metrics import exact_match, shape_of, structural_match


def test_exact_match_normalises():
    assert exact_match("ls  -la;", "ls -la")
    assert not exact_match("ls -l", "ls -la")


def test_shape_ignores_values_and_sudo():
    a = shape_of("sudo find /tmp -type f -name '*.log'")
    b = shape_of("find /var -name foo -type d")
    assert a == b
    assert a.utilities == ("find",)
    assert a.flags == (frozenset({"-type", "-name"}),)


def test_structural_match_tracks_pipeline_order():
    assert structural_match("find . -name '*.py' | wc -l", "find /src -name x | wc -l")
    assert not structural_match("find . | wc -l", "wc -l | find .")
    assert not structural_match("ls -la", "ls -l")


def test_negative_numbers_are_not_flags():
    assert shape_of("tail -n -5 f").flags == (frozenset({"-n"}),)
