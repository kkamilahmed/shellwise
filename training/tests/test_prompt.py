from shellwise_train.prompt import build_messages, build_user_turn


def test_user_turn_without_hint():
    assert build_user_turn("list files") == "request: list files"


def test_user_turn_with_hint():
    assert build_user_turn("anything on port 3000?", "lsof") == (
        "tool: lsof\nrequest: anything on port 3000?"
    )


def test_messages_have_system_then_user():
    msgs = build_messages("x", "ls")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].startswith("tool: ls\n")
