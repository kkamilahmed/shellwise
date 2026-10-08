# shellwise

Natural language to shell command, running fully offline on a laptop CPU.

A LoRA fine-tune of Qwen2.5-0.5B does the translation.
A custom Rust engine with 4-bit weights, a KV cache and SIMD matmuls serves it with no ML dependencies.
Grammar-constrained decoding guarantees every output is a valid command, and a plugin system adds new CLI tools without retraining.

## Layout

| Path | Purpose |
| --- | --- |
| `training/` | Python package: dataset build, LoRA fine-tune, evaluation, export (uv project) |
| `data/` | Built `train/val/test.jsonl` (gitignored) and the committed `manifest.json` |
| `engine/` | Rust inference engine (not started yet) |

## Training pipeline

All commands run from `training/`.

```sh
uv sync
uv run shellwise-build-data --out ../data --cache ../hf-cache
uv run shellwise-train --data ../data --out ../outputs/lora
uv run shellwise-eval --adapter ../outputs/lora --data ../data
uv run shellwise-export --adapter ../outputs/lora --out ../outputs/merged
```

`shellwise-build-data` downloads six public NL-to-shell corpora, normalises them, drops anything that is not a single parseable command line, dedups on (request, command), and hashes the request text into splits so the same request can never leak across them.
The result is about 30K pairs. Sources and licenses are listed in `data/manifest.json`.

`shellwise-train` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA rank 16 on all attention and MLP projections.
Loss is masked to the assistant turn only.
It picks CUDA, then MPS, then CPU, and uses bf16 only on CUDA.

`shellwise-eval` greedy-decodes the test split and reports two accuracies: `exact` (normalised string match) and `structural` (same utilities in the same order with the same flag sets, argument values ignored).

`shellwise-export` merges the adapter into the base weights and writes fp16 safetensors plus `tokenizer.json`.
This directory is the engine's input. Quantisation to 4-bit happens on the engine side.

## Prompt format

The engine must reproduce this exactly. It is defined once in `training/shellwise_train/prompt.py` and rendered through the Qwen2.5 chat template:

```
<|im_start|>system
You translate natural language into a single POSIX shell command. Reply with only the command.<|im_end|>
<|im_start|>user
{request}<|im_end|>
<|im_start|>assistant
```

The model emits the command followed by `<|im_end|>`.

## Development

```sh
cd training
uv run ruff check . && uv run ruff format --check .
uv run pytest
```
