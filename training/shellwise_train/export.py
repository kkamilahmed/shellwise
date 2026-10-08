"""Merge the LoRA adapter into the base weights and save safetensors + tokenizer.

Usage: ``shellwise-export --adapter ../outputs/lora --out ../outputs/merged``

The Rust engine consumes this directory: ``model.safetensors`` (fp16 or fp32),
``config.json`` and ``tokenizer.json``. Quantisation to 4-bit happens on the
engine side so the merged checkpoint stays the single full-precision source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import DEFAULT_MODEL


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("../outputs/merged"))
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args(argv)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
