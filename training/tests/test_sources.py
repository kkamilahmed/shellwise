from shellwise_train.sources import SOURCES, parse_jawa_messages


def test_parse_jawa_messages_strips_prefix():
    raw = (
        "[{'role': 'user', 'content': 'Write the Linux bash command for this task:\\n"
        "Count lines in f'}, {'role': 'assistant', 'content': 'wc -l f'}]"
    )
    assert parse_jawa_messages(raw) == ("Count lines in f", "wc -l f")


def test_parse_jawa_messages_rejects_garbage():
    assert parse_jawa_messages("not a list") is None


def test_source_names_unique():
    names = [s.name for s in SOURCES]
    assert len(names) == len(set(names))
