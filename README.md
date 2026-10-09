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
| `router/` | Python package: two-stage tool router built on Laya, plus the tool catalog (uv project) |
| `scripts/` | `gpu-run.sh`, the one-command reproduction script |
| `engine/` | Rust inference engine (not started yet) |

## Reproduce on a GPU machine

On a fresh Linux box with an NVIDIA driver, any Mac, or Windows from Git Bash, one script runs the whole pipeline.
On Windows the projects pull the CUDA build of PyTorch from PyTorch's own index, since the PyPI wheels there are CPU-only, and the setup phase refuses to continue if torch cannot see a GPU.

```sh
git clone https://github.com/kkamilahmed/shellwise && cd shellwise
./scripts/gpu-run.sh all
```

It installs uv, creates both Python environments, downloads the base models, builds the data, trains the LLM and the router, evaluates both and exports the merged model.
Phases can run on their own, for example `./scripts/gpu-run.sh setup data` then `./scripts/gpu-run.sh train-router`.
Epochs and batch sizes are environment variables documented at the top of the script.
Results land in `outputs/eval/metrics-test.json` and `outputs/eval-router.json`, with logs in `outputs/logs/`.

Defaults fit a 10 GB card.
Rough wall time for the default 4 epochs each: an RTX 3080 takes about 2 hours in total, an M1 Pro about 14 hours.

## Training pipeline

All commands run from `training/`.

```sh
uv sync
uv run shellwise-build-data --out ../data --cache ../hf-cache
uv run shellwise-train --data ../data --out ../outputs/lora
uv run shellwise-eval --adapter ../outputs/lora --data ../data
uv run shellwise-export --adapter ../outputs/lora --out ../outputs/merged
```

`shellwise-build-data` downloads five public NL-to-shell corpora, normalises them, drops anything that is not a single parseable command line, dedups on (request, command), and hashes the request text into splits so the same request can never leak across them.
Each row also carries `tool`, the first utility of its command, which training uses as the router hint.
The result is about 25K pairs. Sources and licenses are listed in `data/manifest.json`.

`shellwise-train` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA rank 16 on all attention and MLP projections.
Loss is masked to the assistant turn only.
70% of examples carry the `tool:` hint the router would provide, so the model learns to follow it, and 30% do not, so it still works when the router abstains.
tldr rows always carry it because their descriptions are unanswerable without the tool.
It picks CUDA, then MPS, then CPU, and uses bf16 only on CUDA.

`shellwise-eval` greedy-decodes the test split and reports two accuracies: `exact` (normalised string match) and `structural` (same utilities in the same order with the same flag sets, argument values ignored).
`--hint reference` feeds each row's own tool as the hint, which measures the model under a perfect router; the default measures it alone.

`shellwise-export` merges the adapter into the base weights and writes fp16 safetensors plus `tokenizer.json`.
This directory is the engine's input. Quantisation to 4-bit happens on the engine side.

## Tool router

Picking the right tool is the hardest part of the translation, so it is a separate step.
The router uses [Laya](https://github.com/NandhaKishorM/laya), a local non-autoregressive decision model that answers typed multiple-choice questions in one forward pass with calibrated confidence.

Routing is two Laya calls.
The first picks one of 13 buckets such as files, text, network or process.
The second picks a tool among that bucket's common tools, with an explicit "other" option so the router can abstain.
Both answers carry a confidence, and below a threshold the request falls through to the language model's free generation.

The catalog in `router/catalog.json` lists every executable on a stock macOS install plus the zsh builtins, 1,341 tools in all, each with its man-page one-liner and a bucket.
Buckets come from a curated table for the roughly 350 tools people actually type, then keyword rules on the description, then a default of `internal` for daemons and helpers.
Only tools marked `common` are offered in stage two.
A plugin registers a tool by adding it to the catalog with a description, which is also how the long tail becomes routable without retraining anything.

```sh
cd router
uv sync
uv run shellwise-build-catalog --out catalog.json --train ../data/train.jsonl
uv run shellwise-route "is there anything running on port 3000"
uv run shellwise-eval-router --data ../data/test.jsonl --per-tool 8
```

## Prompt format

The engine must reproduce this exactly. It is defined once in `training/shellwise_train/prompt.py` and rendered through the Qwen2.5 chat template:

```
<|im_start|>system
You translate natural language into a single POSIX shell command. Reply with only the command.<|im_end|>
<|im_start|>user
tool: {tool}
request: {request}<|im_end|>
<|im_start|>assistant
```

The `tool:` line is present when the router is confident and omitted when it abstains; the `request:` prefix is always there.
The model emits the command followed by `<|im_end|>`.

## Development

```sh
cd training
uv run ruff check . && uv run ruff format --check .
uv run pytest
```
