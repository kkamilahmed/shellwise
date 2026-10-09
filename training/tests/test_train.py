import random

from shellwise_train.train import hint_for


def test_tldr_rows_always_get_the_hint():
    rng = random.Random(0)
    rows = [{"source": "tldr", "tool": "pio"}] * 200
    assert all(hint_for(r, rng) == "pio" for r in rows)


def test_other_rows_get_the_hint_most_of_the_time():
    rng = random.Random(0)
    rows = [{"source": "nl2bash", "tool": "find"}] * 2000
    rate = sum(hint_for(r, rng) is not None for r in rows) / len(rows)
    assert 0.65 < rate < 0.75


def test_missing_tool_means_no_hint():
    assert hint_for({"source": "nl2bash"}, random.Random(0)) is None
