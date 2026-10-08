"""LoRA fine-tune of Qwen2.5-0.5B on the shellwise dataset.

Usage: ``shellwise-train --data ../data --out ../outputs/lora``

Loss is computed only on the assistant turn (the command plus end-of-turn).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .common import pick_device, pick_dtype, read_jsonl
from .prompt import render_completion, render_prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    data: Path = Path("../data")
    out: Path = Path("../outputs/lora")
    epochs: float = 2.0
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    batch: int = 8
    grad_accum: int = 2
    max_len: int = 192
    max_steps: int = -1
    limit: int | None = None
    seed: int = 13


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.items = []
        truncated = 0
        for row in rows:
            prompt_ids = tokenizer(render_prompt(tokenizer, row["nl"]), add_special_tokens=False)[
                "input_ids"
            ]
            completion_ids = tokenizer(
                render_completion(tokenizer, row["cmd"]), add_special_tokens=False
            )["input_ids"]
            ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            if len(ids) > max_len:
                truncated += 1
                ids, labels = ids[:max_len], labels[:max_len]
            self.items.append((ids, labels))
        if truncated:
            print(f"warning: truncated {truncated}/{len(rows)} examples to {max_len} tokens")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


class Collator:
    """Right-pads to a multiple of ``pad_to`` so the allocator sees few distinct shapes.

    On MPS every new tensor shape grows the cached pool, and length-grouped
    batches would otherwise produce a different shape on almost every step.
    """

    def __init__(self, pad_id: int, pad_to: int = 32):
        self.pad_id = pad_id
        self.pad_to = pad_to

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(b["input_ids"]) for b in batch)
        width = math.ceil(longest / self.pad_to) * self.pad_to
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class ReleaseMpsCache(TrainerCallback):
    """Hand cached MPS blocks back to the OS so the process stays inside physical RAM."""

    def __init__(self, every: int = 10):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % self.every == 0:
            torch.mps.empty_cache()


def run(cfg: Config) -> None:
    device, dtype = pick_device(), pick_dtype(pick_device())
    print(f"device={device} dtype={dtype} model={cfg.model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_rows = read_jsonl(cfg.data / "train.jsonl", cfg.limit)
    val_rows = read_jsonl(cfg.data / "val.jsonl", cfg.limit)
    train_ds = PairDataset(train_rows, tokenizer, cfg.max_len)
    val_ds = PairDataset(val_rows, tokenizer, cfg.max_len)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch * cfg.grad_accum))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, round(total_steps * cfg.warmup_ratio))

    args = TrainingArguments(
        output_dir=str(cfg.out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=cfg.batch,
        per_device_eval_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=dtype == torch.bfloat16,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        seed=cfg.seed,
        dataloader_pin_memory=device.type == "cuda",
        remove_unused_columns=False,
        train_sampling_strategy="group_by_length",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[ReleaseMpsCache()],
    )
    trainer.train()

    cfg.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.out)
    tokenizer.save_pretrained(cfg.out)
    (cfg.out / "train_config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, indent=2
        )
        + "\n"
    )
    print(f"saved adapter to {cfg.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for field, value in asdict(defaults).items():
        kind = type(getattr(defaults, field)) if value is not None else int
        ap.add_argument(f"--{field.replace('_', '-')}", type=kind, default=value)
    ns = ap.parse_args(argv)
    run(Config(**vars(ns)))


if __name__ == "__main__":
    main()
