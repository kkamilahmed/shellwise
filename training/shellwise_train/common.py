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
    # bf16 is only reliable on CUDA; MPS and CPU train/infer in fp32 at this model size.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows
