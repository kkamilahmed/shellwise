"""Fine-tune the Laya English checkpoint on the two routing questions.

Usage: ``shellwise-finetune-router --data finetune --out ../outputs/laya-router``

Wraps ``laya.train.finetune`` with the shellwise defaults. ``--limit`` trains
on a prefix of the rows, which is how to time a run before committing to it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LAYA_REPO = "convaiinnovations/laya"


def local_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        LAYA_REPO, allow_patterns=["*.json", "*.safetensors", "tokenizer/*", "encoder/*"]
    )


def _patch_calibration_device(laya_train) -> None:
    """laya 0.4.0 evaluates the base checkpoint before ``train_model`` moves it to the
    device, which fails on MPS with "Passed CPU tensor to MPS op". Move it first."""
    original = laya_train.calibration_records

    def patched(model, tok, items, device, *args, **kwargs):
        model.to(device)
        return original(model, tok, items, device, *args, **kwargs)

    laya_train.calibration_records = patched


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("finetune"))
    ap.add_argument("--out", type=Path, default=Path("../outputs/laya-router"))
    ap.add_argument("--base", default=None, help="local checkpoint dir; default: hub cache")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--limit", type=int, help="train on the first N rows only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freeze-encoder", action="store_true")
    args = ap.parse_args(argv)

    import laya.train as laya_train
    from laya.train import TrainConfig, finetune

    _patch_calibration_device(laya_train)

    train_path = args.data / "train.jsonl"
    if args.limit:
        subset = args.out / "train-subset.jsonl"
        args.out.mkdir(parents=True, exist_ok=True)
        with train_path.open() as src, subset.open("w") as dst:
            for i, line in enumerate(src):
                if i >= args.limit:
                    break
                dst.write(line)
        train_path = subset

    cfg = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        freeze_encoder=args.freeze_encoder,
        eval_data=str(args.data / "val.jsonl"),
        log_every=20,
    )
    base = args.base or local_checkpoint()
    print(f"base={base}")
    print(f"train={train_path} epochs={cfg.epochs} batch={cfg.micro_batch}x{cfg.grad_accum}")
    t0 = time.time()
    summary = finetune(str(train_path), base, str(args.out), cfg, device=args.device)
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out / "finetune_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
