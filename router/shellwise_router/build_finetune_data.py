"""Turn shellwise pairs into Laya fine-tuning rows for the two routing questions.

Usage: ``shellwise-build-router-data --catalog catalog.json --data ../data --out finetune``

Each row carries the stage-one bucket question and the stage-two tool question
for the reference tool's bucket, with ``expected`` answers for both. Rows whose
reference tool is not a common routable catalog tool are skipped, and the
dominant tool (find) is capped so the router does not learn to always say find.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .buckets import ROUTABLE
from .build_catalog import first_util
from .router import OTHER, bucket_criteria, tool_criteria


def build_rows(pairs: list[dict], tools: dict, cap: int, rng: random.Random) -> list[dict]:
    rng.shuffle(pairs)
    per: Counter = Counter()
    routable = {b.id for b in ROUTABLE}
    rows = []
    for p in pairs:
        t = first_util(p["cmd"])
        e = tools.get(t)
        if not e or not e["common"] or e["bucket"] not in routable:
            continue
        if per[t] >= cap:
            continue
        per[t] += 1
        rows.append(
            {
                "state": {"request": p["nl"]},
                "questions": {
                    "bucket": {
                        "type": "choice",
                        "instructions": "Which area does this shell request belong to?",
                        "criteria": bucket_criteria(),
                    },
                    "tool": {
                        "type": "choice",
                        "instructions": "Which command-line tool does this request need?",
                        "criteria": tool_criteria(tools, e["bucket"]),
                    },
                },
                "expected": {"bucket": e["bucket"], "tool": t},
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("finetune"))
    ap.add_argument("--cap", type=int, default=800, help="max rows per reference tool")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    tools = json.loads(args.catalog.read_text())["tools"]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        pairs = [json.loads(line) for line in (args.data / f"{split}.jsonl").open()]
        rows = build_rows(pairs, tools, args.cap if split == "train" else 10**9, rng)
        with (args.out / f"{split}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        buckets = Counter(r["expected"]["bucket"] for r in rows)
        print(f"{split}: {len(rows)} rows from {len(pairs)} pairs")
        print(f"  buckets: {dict(buckets.most_common())}")
    assert OTHER  # the tool question always carries the abstain option


if __name__ == "__main__":
    main()
