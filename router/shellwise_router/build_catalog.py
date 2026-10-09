"""Build ``catalog.json``: every tool on the machine with a description and a bucket.

Usage: ``shellwise-build-catalog --out catalog.json [--train ../data/train.jsonl]``

Sources, in priority order: the curated table, keyword rules on the whatis
description, then a default of ``internal`` for section-8 daemons and
``shell`` for uncurated zsh builtins. Everything else lands in ``internal``
too, which is the safe default: a tool nobody asked for should not be offered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

from .buckets import BUCKET_IDS
from .curated import CURATED

STOCK_DIRS = ("/bin", "/sbin", "/usr/bin", "/usr/sbin")
EXTRA_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

# (regex over the lowercase whatis description, bucket). First match wins.
KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    (r"daemon|agent\b|helper|server for|xpc|launchd job|background", "internal"),
    (r"compress|archive|\btar\b|\bzip\b", "archive"),
    (r"\bprint(er|ing| queue| job)|cups|\bfont", "media"),
    (r"image|audio|sound|video|movie|pdf|rtf|document conver", "media"),
    (r"sqlite|property list|plist|json|xml|base64|checksum|digest|hash|uuid|hex", "data"),
    (r"\bssh\b|\bhttp|url|dns|network|socket|\bport\b|interface|wifi|bluetooth|packet", "network"),
    (r"process|signal|\bjob\b|scheduler|cpu usage|memory usage", "process"),
    (r"disk|volume|partition|mount|filesystem|file system|image \(dmg", "disk"),
    (r"\buser\b|group|password|login|certificate|keychain|sign|encrypt|crypt", "users"),
    (
        r"compil|link editor|linker|debugg|assembl|object file|mach-o|xcode"
        r"|swift|python|perl|ruby|git",
        "dev",
    ),
    (r"text|line|word|pattern|regular expression|sort|column|stream editor|edit", "text"),
    (r"file|director|path|symbolic link", "files"),
    (r"kernel|system|boot|power|sleep|time|date|hardware|log\b", "system"),
    (r"shell|alias|environment|variable|history", "shell"),
)
_RULES = tuple((re.compile(p), b) for p, b in KEYWORD_RULES)


def list_tools(dirs: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for d in dirs:
        if os.path.isdir(d):
            out |= {n for n in os.listdir(d) if not n.startswith(".")}
    return out


def zsh_builtins() -> set[str]:
    r = subprocess.run(["zsh", "-c", "print -l ${(k)builtins}"], capture_output=True, text=True)
    return set(r.stdout.split())


_ENTRY = re.compile(r"^(?:/\S+/)?([^\s(]+)\((\w+)\)$")
_LINE = re.compile(r"^(.*?)\s+-\s+(.*)$")


def whatis_descriptions(tools: set[str]) -> dict[str, tuple[str, str]]:
    """tool -> (man section, one-line description), preferring sections 1 then 8."""
    dump = subprocess.run(["apropos", "."], capture_output=True, text=True, errors="replace")
    best: dict[str, tuple[int, str, str]] = {}
    for line in dump.stdout.splitlines():
        m = _LINE.match(line.rstrip())
        if not m:
            continue
        names, desc = m.groups()
        for ent in names.split(","):
            mm = _ENTRY.match(ent.strip())
            if not mm:
                continue
            name, sec = mm.groups()
            if name not in tools:
                continue
            prio = {"1": 0, "8": 1, "1m": 1}.get(sec.lower(), 5)
            if name not in best or prio < best[name][0]:
                best[name] = (prio, sec, desc.strip())
    return {k: (sec, desc) for k, (_, sec, desc) in best.items()}


def assign_bucket(name: str, section: str | None, desc: str | None, builtin: bool) -> str:
    if name in CURATED:
        return CURATED[name]
    if desc:
        low = desc.lower()
        for rx, b in _RULES:
            if rx.search(low):
                return b
    if builtin:
        return "shell"
    return "internal"


def first_util(cmd: str) -> str | None:
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return None
    if toks and toks[0] == "sudo":
        toks = toks[1:]
    return toks[0] if toks else None


def train_counts(path: Path | None) -> Counter:
    c: Counter = Counter()
    if path and path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                u = first_util(json.loads(line)["cmd"])
                if u:
                    c[u] += 1
    return c


def build(train: Path | None, include_extra: bool, common_min: int) -> dict:
    stock = list_tools(STOCK_DIRS)
    extra = list_tools(EXTRA_DIRS) - stock if include_extra else set()
    builtins = zsh_builtins()
    tools = stock | extra | builtins
    descs = whatis_descriptions(tools)
    counts = train_counts(train)

    entries = {}
    for name in sorted(tools):
        sec, desc = descs.get(name, (None, None))
        b = assign_bucket(name, sec, desc, name in builtins)
        entries[name] = {
            "bucket": b,
            "description": desc,
            "stock": name in stock or name in builtins,
            "builtin": name in builtins,
            "train_examples": counts.get(name, 0),
            "common": name in CURATED or counts.get(name, 0) >= common_min,
        }
    assert all(e["bucket"] in BUCKET_IDS for e in entries.values())
    summary = Counter(e["bucket"] for e in entries.values())
    common = Counter(e["bucket"] for e in entries.values() if e["common"])
    return {
        "tools": entries,
        "summary": {b: {"total": summary[b], "common": common[b]} for b in BUCKET_IDS},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("catalog.json"))
    ap.add_argument("--train", type=Path, default=Path("../data/train.jsonl"))
    ap.add_argument("--include-extra", action="store_true", help="also scan Homebrew dirs")
    ap.add_argument("--common-min", type=int, default=5)
    args = ap.parse_args(argv)
    cat = build(args.train, args.include_extra, args.common_min)
    args.out.write_text(json.dumps(cat, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(cat["summary"], indent=1))
    print(f"{len(cat['tools'])} tools -> {args.out}")


if __name__ == "__main__":
    main()
