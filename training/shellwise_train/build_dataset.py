"""Build train/val/test JSONL from the public sources.

Usage: ``shellwise-build-data --out ../data``

Dedup is on (nl key, normalised cmd). Splits are assigned by hashing the NL
key, so the same request can never appear in two splits even with different
reference commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import is_valid_pair, nl_key, normalize_cmd, normalize_nl
from .sources import SOURCES, Pair, download


def _split_for(key: str, val_frac: float, test_frac: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def collect(cache_dir: Path, only: set[str] | None = None) -> tuple[list[Pair], Counter]:
    dropped: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    kept: list[Pair] = []
    for source in SOURCES:
        if only and source.name not in only:
            continue
        files = download(source, cache_dir)
        for pair in source.load(files):
            nl, cmd = normalize_nl(pair.nl), normalize_cmd(pair.cmd)
            if not is_valid_pair(nl, cmd):
                dropped[f"{source.name}:invalid"] += 1
                continue
            key = (nl_key(nl), cmd)
            if key in seen:
                dropped[f"{source.name}:duplicate"] += 1
                continue
            seen.add(key)
            kept.append(Pair(nl, cmd, source.name))
    return kept, dropped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("../data"))
    ap.add_argument("--cache", type=Path, default=Path("../hf-cache"))
    ap.add_argument("--sources", nargs="*", help="subset of source names")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    pairs, dropped = collect(args.cache, set(args.sources) if args.sources else None)
    splits: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        splits[_split_for(nl_key(pair.nl), args.val_frac, args.test_frac, args.seed)].append(pair)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for p in rows:
                fh.write(json.dumps({"nl": p.nl, "cmd": p.cmd, "source": p.source}) + "\n")

    manifest = {
        "total": len(pairs),
        "splits": {k: len(v) for k, v in splits.items()},
        "per_source": dict(Counter(p.source for p in pairs)),
        "dropped": dict(dropped),
        "sources": [
            {"name": s.name, "repo": s.repo, "license": s.license}
            for s in SOURCES
            if not args.sources or s.name in args.sources
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
