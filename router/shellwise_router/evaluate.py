"""Score the two-stage router on the shellwise test split.

Usage: ``shellwise-eval-router --data ../data/test.jsonl --per-tool 8``

Rows whose reference tool is not a common catalog tool are skipped, since
the router cannot pick what it does not offer. ``--per-tool`` caps rows per
reference tool so find does not dominate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build_catalog import first_util
from .router import ToolRouter


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("../data/test.jsonl"))
    ap.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    ap.add_argument("--per-tool", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path, default=Path("eval-router.json"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    router = ToolRouter(args.catalog, device=args.device, bucket_threshold=0.0, tool_threshold=0.0)
    rows = [json.loads(line) for line in args.data.open()]
    rows = [
        r for r in rows if (u := first_util(r["cmd"])) in router.tools and router.tools[u]["common"]
    ]
    random.Random(args.seed).shuffle(rows)
    per: Counter = Counter()
    sample = []
    for r in rows:
        u = first_util(r["cmd"])
        if per[u] < args.per_tool:
            per[u] += 1
            sample.append(r)
    if args.limit:
        sample = sample[: args.limit]
    print(f"evaluating {len(sample)} rows over {len(per)} reference tools")

    stats: Counter = Counter()
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    confusions: Counter = Counter()
    lat = []
    records = []
    for i, r in enumerate(sample, 1):
        ref = first_util(r["cmd"])
        ref_bucket = router.tools[ref]["bucket"]
        res = router.route(r["nl"])
        lat.append(res.ms)
        b_ok = res.bucket == ref_bucket
        best = max(res.tool_probs, key=res.tool_probs.get) if res.tool_probs else None
        t_ok = best == ref
        stats["n"] += 1
        stats["bucket"] += b_ok
        stats["tool"] += t_ok
        stats["tool_given_bucket"] += t_ok and b_ok
        by_bucket[ref_bucket]["n"] += 1
        by_bucket[ref_bucket]["bucket"] += b_ok
        by_bucket[ref_bucket]["tool"] += t_ok
        if not t_ok:
            confusions[(ref, res.bucket + "/" + str(best))] += 1
        records.append(
            {
                **r,
                "ref_tool": ref,
                "ref_bucket": ref_bucket,
                "bucket": res.bucket,
                "bucket_conf": res.bucket_confidence,
                "tool": best,
                "tool_conf": res.tool_confidence,
                "ok": t_ok,
            }
        )
        print(
            f"  {i}/{len(sample)}  bucket {stats['bucket'] / i:.2f}  tool {stats['tool'] / i:.2f}",
            end="\r",
        )
    print()
    lat.sort()
    n = stats["n"]
    report = {
        "n": n,
        "bucket_accuracy": round(stats["bucket"] / n, 4),
        "tool_accuracy": round(stats["tool"] / n, 4),
        "bucket_right_and_tool_right": round(stats["tool_given_bucket"] / n, 4),
        "latency_ms": {"p50": lat[len(lat) // 2], "p90": lat[int(len(lat) * 0.9)]},
        "per_bucket": {
            b: {
                "n": c["n"],
                "bucket": round(c["bucket"] / c["n"], 3),
                "tool": round(c["tool"] / c["n"], 3),
            }
            for b, c in sorted(by_bucket.items())
        },
        "top_confusions": [f"{a} -> {b} x{k}" for (a, b), k in confusions.most_common(12)],
    }
    args.out.write_text(json.dumps({"report": report, "records": records}, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
