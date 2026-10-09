"""Helpers shared by train / eval / export."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """bf16 on GPUs that support it natively (Ampere and newer); fp32 elsewhere.

    Pre-Ampere cards emulate bf16 slowly, and MPS/CPU are safest in fp32 at this
    model size, so anything else gets fp32.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows
