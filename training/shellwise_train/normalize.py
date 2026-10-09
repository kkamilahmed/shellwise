"""Command/NL normalisation, filtering and dedup keys shared by build and eval."""

from __future__ import annotations

import re
import shlex

PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}")
# A bare ``\`` token is never valid shell; it shows up where a source corpus lost ``\(``.
LONE_BACKSLASH_RE = re.compile(r"(^|\s)\\(\s|$)")
# A ``$`` followed by whitespace or end of line, outside quotes, is a command
# substitution whose ``(...)`` a source corpus stripped: ``cd $``, ``arr=$``.
DANGLING_DOLLAR_RE = re.compile(r"""(^|[^"'\\])\$(\s|$)""")
MAX_CMD_CHARS = 200
MAX_NL_CHARS = 300
MIN_NL_WORDS = 2


def strip_placeholders(cmd: str) -> str:
    """tldr style ``{{path/to/file}}`` -> ``path/to/file``."""
    return PLACEHOLDER_RE.sub(r"\1", cmd)


def normalize_cmd(cmd: str) -> str:
    """Canonical form used for dedup and exact-match scoring.

    Collapses whitespace runs (also inside quotes, which is acceptable because
    both prediction and reference go through the same function) and drops a
    trailing semicolon.
    """
    cmd = strip_placeholders(cmd).strip()
    cmd = " ".join(cmd.split())
    # Drop a trailing statement terminator, but keep the ``\;`` that ends find -exec.
    if cmd.endswith(";") and not cmd.endswith("\\;"):
        cmd = cmd[:-1].rstrip()
    return cmd


def normalize_nl(nl: str) -> str:
    nl = " ".join(nl.split()).strip()
    return nl[:1].upper() + nl[1:] if nl else nl


def nl_key(nl: str) -> str:
    """Case/punctuation-insensitive key used for dedup and split assignment."""
    return re.sub(r"[^a-z0-9 ]+", "", normalize_nl(nl).lower()).strip()


def clean_completion(text: str) -> str | None:
    """Reduce a model-style completion to one command line, or None.

    Drops shebang scripts, fenced code markers, and ``# comment`` lines such as
    the sample output some datasets append after the command.
    """
    lines = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        lines.append(line)
    if len(lines) != 1:
        return None
    return lines[0]


def strip_trailing_output(cmd: str) -> str:
    """Cut at the first literal ``\\n`` that sits outside quotes.

    Some generated corpora append sample output after the command as
    ``cmd\\n output...``. A ``\\n`` inside quotes (``tr ':' '\\n'``) is kept.
    """
    quote: str | None = None
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"':
                i += 1
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "\\":
            if cmd[i + 1 : i + 2] == "n":
                return cmd[:i].rstrip()
            i += 1
        i += 1
    return cmd


def is_valid_pair(nl: str, cmd: str) -> bool:
    if not nl or not cmd:
        return False
    if "\n" in cmd or len(cmd) > MAX_CMD_CHARS or len(nl) > MAX_NL_CHARS:
        return False
    if len(nl.split()) < MIN_NL_WORDS:
        return False
    if cmd.startswith("#") or "{{" in cmd:
        return False
    if not cmd.isascii() or LONE_BACKSLASH_RE.search(cmd) or DANGLING_DOLLAR_RE.search(cmd):
        return False
    try:
        shlex.split(cmd)
    except ValueError:
        return False
    return True


def first_utility(cmd: str) -> str | None:
    """The program a command line runs first: ``sudo lsof -i :3000`` -> ``lsof``.

    Skips ``sudo`` and leading ``NAME=value`` assignments. Returns None when the
    line does not tokenise.
    """
    try:
        toks = shlex.split(normalize_cmd(cmd))
    except ValueError:
        return None
    for tok in toks:
        if tok == "sudo":
            continue
        if "=" in tok and tok.split("=", 1)[0].isidentifier():
            continue
        return tok
    return None
