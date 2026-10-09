"""Greedy-decode the test split and report accuracy.

Usage: ``shellwise-eval --adapter ../outputs/lora --data ../data``
Pass ``--adapter`` for a LoRA directory or ``--merged`` for an exported model.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import pick_device, pick_dtype, read_jsonl
from .metrics import exact_match, structural_match
from .prompt import render_prompt
from .train import DEFAULT_MODEL


def load_model(base: str, adapter: Path | None, merged: Path | None, device, dtype):
    if merged:
        tok = AutoTokenizer.from_pretrained(merged)
        model = AutoModelForCausalLM.from_pretrained(merged, dtype=dtype)
    else:
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(adapter or base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    tok.padding_side = "left"
    return tok, model.to(device).eval()


@torch.no_grad()
def generate(tok, model, nls: list[str], max_new_tokens: int, batch: int, device) -> list[str]:
    outs: list[str] = []
    for i in range(0, len(nls), batch):
        prompts = [render_prompt(tok, nl) for nl in nls[i : i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :]
        outs.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
        print(f"  {min(i + batch, len(nls))}/{len(nls)}", end="\r", flush=True)
    print()
    return outs


def score(rows: list[dict], preds: list[str]) -> dict:
    totals: Counter = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    for row, pred in zip(rows, preds, strict=True):
        ex, st = exact_match(pred, row["cmd"]), structural_match(pred, row["cmd"])
        for bucket in (totals, per_source[row["source"]]):
            bucket["n"] += 1
            bucket["exact"] += ex
            bucket["structural"] += st
    pct = lambda c: {  # noqa: E731
        "n": c["n"],
        "exact": round(c["exact"] / c["n"], 4),
        "structural": round(c["structural"] / c["n"], 4),
    }
    return {"overall": pct(totals), "per_source": {k: pct(v) for k, v in per_source.items()}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--merged", type=Path)
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("../outputs/eval"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args(argv)

    device = pick_device()
    tok, model = load_model(args.base, args.adapter, args.merged, device, pick_dtype(device))
    rows = read_jsonl(args.data / f"{args.split}.jsonl", args.limit)
    print(f"evaluating {len(rows)} rows on {device}")

    t0 = time.time()
    preds = generate(tok, model, [r["nl"] for r in rows], args.max_new_tokens, args.batch, device)
    elapsed = time.time() - t0

    report = score(rows, preds)
    report["seconds"] = round(elapsed, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"predictions-{args.split}.jsonl").open("w", encoding="utf-8") as fh:
        for row, pred in zip(rows, preds, strict=True):
            fh.write(
                json.dumps(
                    {
                        **row,
                        "pred": pred,
                        "exact": exact_match(pred, row["cmd"]),
                        "structural": structural_match(pred, row["cmd"]),
                    }
                )
                + "\n"
            )
    (args.out / f"metrics-{args.split}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
