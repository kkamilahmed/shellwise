from shellwise_train.normalize import (
    clean_completion,
    first_utility,
    is_valid_pair,
    nl_key,
    normalize_cmd,
    strip_placeholders,
    strip_trailing_output,
)


def test_strip_placeholders():
    assert strip_placeholders("git commit -m {{message}}") == "git commit -m message"
    assert strip_placeholders("cp {{a}} {{b}}") == "cp a b"


def test_normalize_cmd_collapses_and_strips_semicolon():
    assert normalize_cmd("  ls   -la ;") == "ls -la"
    assert normalize_cmd("find . -name '*.py' | wc -l") == "find . -name '*.py' | wc -l"


def test_nl_key_is_case_and_punctuation_insensitive():
    assert nl_key("List all files!") == nl_key("list all files")


def test_clean_completion_rejects_scripts_and_strips_comment_output():
    assert clean_completion("grep -R TODO . | wc -l\n# 27") == "grep -R TODO . | wc -l"
    assert clean_completion("#!/usr/bin/env bash\necho hi") == "echo hi"  # single real line
    assert clean_completion("a=1\necho $a") is None
    assert clean_completion("```bash\nls\n```") == "ls"


def test_is_valid_pair():
    assert is_valid_pair("List files", "ls -la")
    assert not is_valid_pair("ls", "ls -la")  # too short
    assert not is_valid_pair("List files", "ls {{dir}}")
    assert not is_valid_pair("List files", "echo 'unterminated")
    assert not is_valid_pair("List files", "x" * 201)


def test_find_exec_terminator_survives_normalisation():
    assert (
        normalize_cmd("find . -type d -exec chmod +x {} \\;")
        == "find . -type d -exec chmod +x {} \\;"
    )
    assert is_valid_pair("Make dirs executable", "find . -type d -exec chmod +x {} \\;")


def test_rejects_corrupted_and_non_ascii_commands():
    assert not is_valid_pair("Find things", "find / \\ -prune -o \\ -print")
    assert not is_valid_pair("Find things", "find . -newermt \u201cSep 1\u201d")
    assert is_valid_pair("Escape a space", "ls /media/2TB\\ Data")


def test_strip_trailing_output_respects_quotes():
    assert strip_trailing_output("echo $PATH | tr ':' '\\n'") == "echo $PATH | tr ':' '\\n'"
    assert strip_trailing_output('printf "a\\nb"') == 'printf "a\\nb"'
    assert strip_trailing_output("openssl rand -hex 8\\nA9f3kLmP") == "openssl rand -hex 8"
    assert strip_trailing_output("df -h\\n Filesystem Size") == "df -h"


def test_first_utility_skips_sudo_and_assignments():
    assert first_utility("sudo lsof -i :3000") == "lsof"
    assert first_utility("FOO=1 BAR=2 make all") == "make"
    assert first_utility("find . -name x | wc -l") == "find"
    assert first_utility("echo 'unterminated") is None


def test_rejects_truncated_command_substitution():
    assert not is_valid_pair("Change dir", "cd $")
    assert not is_valid_pair("Set var", "arr=$")
    assert not is_valid_pair("Find", "find $ -type f")
    assert is_valid_pair("Print path", "echo $PATH")
    assert is_valid_pair("Regex anchor", "grep 'foo$' file")
    assert is_valid_pair("Substitution kept", "echo $(date)")
