"""Scoring for predicted commands.

* ``exact``: normalised string equality.
* ``structural``: same utilities in the same order with the same flag sets.
  Argument values (paths, patterns, numbers) are ignored, so it measures
  whether the model chose the right tools and options.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from .normalize import normalize_cmd

_SEPARATORS = {"|", "||", "&&", ";", "&"}


@dataclass(frozen=True)
class Shape:
    utilities: tuple[str, ...]
    flags: tuple[frozenset[str], ...]


def shape_of(cmd: str) -> Shape | None:
    try:
        tokens = shlex.split(normalize_cmd(cmd))
    except ValueError:
        return None
    if not tokens:
        return None
    utilities: list[str] = []
    flags: list[set[str]] = []
    new_cmd = True
    for tok in tokens:
        if tok in _SEPARATORS:
            new_cmd = True
            continue
        if new_cmd:
            if tok == "sudo":
                continue
            utilities.append(tok)
            flags.append(set())
            new_cmd = False
        elif tok.startswith("-") and len(tok) > 1 and not tok[1:].isdigit():
            flags[-1].add(tok.split("=", 1)[0])
    return Shape(tuple(utilities), tuple(frozenset(f) for f in flags))


def exact_match(pred: str, ref: str) -> bool:
    return normalize_cmd(pred) == normalize_cmd(ref)


def structural_match(pred: str, ref: str) -> bool:
    a, b = shape_of(pred), shape_of(ref)
    return a is not None and a == b
