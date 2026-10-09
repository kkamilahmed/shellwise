#!/usr/bin/env bash
# Reproduce the shellwise training pipeline on a fresh machine.
#
#   git clone https://github.com/kkamilahmed/shellwise && cd shellwise
#   ./scripts/gpu-run.sh all
#
# Phases (run one, several, or "all"):
#   setup         install uv, create both Python environments, download model weights
#   data          build the NL/shell dataset and the router fine-tuning rows
#   train-llm     LoRA fine-tune of Qwen2.5-0.5B-Instruct
#   train-router  fine-tune the Laya tool router
#   eval          score both models on the test split
#   export        merge the LoRA adapter into fp16 safetensors for the engine
#
# Tunables (environment variables):
#   LLM_EPOCHS=4  LLM_BATCH=16  ROUTER_EPOCHS=4  ROUTER_MICRO_BATCH=8  ROUTER_GRAD_ACCUM=4
#   ROUTER_ROWS=   (limit router training rows; empty = all)
#
# Defaults fit a 10 GB card such as an RTX 3080. With 16 GB or more, raise
# ROUTER_MICRO_BATCH to 16 and ROUTER_GRAD_ACCUM to 2 for a faster run.
#
# Needs: git, curl and an NVIDIA driver for GPU runs, on Linux, macOS or
# Windows (run from Git Bash; the CUDA build of torch is selected
# automatically there). Everything else is installed into the repo directory.
#   ALLOW_CPU=1   continue even if torch sees no GPU Outputs land in ./outputs and
# logs in ./outputs/logs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/logs

LLM_EPOCHS="${LLM_EPOCHS:-4}"
LLM_BATCH="${LLM_BATCH:-16}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-4}"
ROUTER_MICRO_BATCH="${ROUTER_MICRO_BATCH:-8}"
ROUTER_GRAD_ACCUM="${ROUTER_GRAD_ACCUM:-4}"
ROUTER_ROWS="${ROUTER_ROWS:-}"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
timed() { # timed <label> <cmd...>
  local label=$1; shift
  local start; start=$(date +%s)
  local file; file=$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '-')
  log "$label"
  "$@" 2>&1 | tee "outputs/logs/${file}.log"
  printf '    %s finished in %d min\n' "$label" $(( ($(date +%s) - start) / 60 ))
}

gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  elif [[ "$(uname)" == "Darwin" ]]; then
    echo "Apple Silicon (MPS): $(sysctl -n machdep.cpu.brand_string)"
  else
    echo "no GPU detected; training will run on CPU and be very slow"
  fi
}

phase_setup() {
  log "machine"
  gpu_info
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || die "uv not on PATH after install; open a new shell and rerun"
  timed "sync training env" bash -c "cd training && uv sync --python 3.12"
  timed "sync router env"   bash -c "cd router && uv sync --python 3.12"
  timed "verify torch sees the GPU" bash -c "cd training && uv run python -c '
import torch, sys
dev = \"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\"
print(\"torch\", torch.__version__, \"device\", dev)
if dev == \"cuda\":
    print(\"gpu\", torch.cuda.get_device_name(0), \"bf16\", torch.cuda.is_bf16_supported())
elif dev == \"cpu\":
    print(\"no GPU visible to torch (build\", torch.__version__ + \"). Training on CPU would take days.\")
    sys.exit(0 if \"${ALLOW_CPU:-}\" else 3)
'" || die "torch cannot see a GPU. If the build above ends in +cpu, rerun ./scripts/gpu-run.sh setup after pulling the latest repo; set ALLOW_CPU=1 to proceed anyway"
  timed "download base models" bash -c "cd training && uv run python -c '
from huggingface_hub import snapshot_download
snapshot_download(\"Qwen/Qwen2.5-0.5B-Instruct\")
snapshot_download(\"convaiinnovations/laya\", allow_patterns=[\"*.json\", \"*.safetensors\", \"tokenizer/*\", \"encoder/*\"])
print(\"ok\")
'"
}

phase_data() {
  timed "build NL/shell dataset" bash -c "cd training && uv run shellwise-build-data --out ../data --cache ../hf-cache"
  # The tool catalog is committed; it describes a stock macOS install and is
  # not rebuilt here because this machine is probably Linux.
  timed "build router fine-tune rows" bash -c "cd router && uv run shellwise-build-router-data --catalog catalog.json --data ../data --out finetune --cap 800"
}

phase_train_llm() {
  timed "train LLM (LoRA, ${LLM_EPOCHS} epochs, batch ${LLM_BATCH})" bash -c \
    "cd training && uv run shellwise-train --data ../data --out ../outputs/lora --epochs ${LLM_EPOCHS} --batch ${LLM_BATCH} --grad-accum 1"
}

phase_train_router() {
  local limit=()
  [[ -n "$ROUTER_ROWS" ]] && limit=(--limit "$ROUTER_ROWS")
  timed "train router (Laya, ${ROUTER_EPOCHS} epochs)" bash -c \
    "cd router && uv run shellwise-finetune-router --data finetune --out ../outputs/laya-router --epochs ${ROUTER_EPOCHS} --micro-batch ${ROUTER_MICRO_BATCH} --grad-accum ${ROUTER_GRAD_ACCUM} ${limit[*]:-}"
}

phase_eval() {
  [[ -d outputs/lora ]] || die "outputs/lora missing; run train-llm first"
  timed "eval LLM on test split" bash -c "cd training && uv run shellwise-eval --adapter ../outputs/lora --data ../data --out ../outputs/eval --batch 64"
  if [[ -d outputs/laya-router ]]; then
    timed "eval router on test split" bash -c "cd router && uv run shellwise-eval-router --data ../data/test.jsonl --catalog catalog.json --per-tool 8 --model ../outputs/laya-router --out ../outputs/eval-router.json"
  else
    echo "outputs/laya-router missing; skipping router eval"
  fi
}

phase_export() {
  [[ -d outputs/lora ]] || die "outputs/lora missing; run train-llm first"
  timed "export merged fp16 model" bash -c "cd training && uv run shellwise-export --adapter ../outputs/lora --out ../outputs/merged"
}

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[[ $# -ge 1 ]] || usage
for phase in "$@"; do
  case "$phase" in
    setup)        phase_setup ;;
    data)         phase_data ;;
    train-llm)    phase_train_llm ;;
    train-router) phase_train_router ;;
    eval)         phase_eval ;;
    export)       phase_export ;;
    all)          phase_setup; phase_data; phase_train_llm; phase_train_router; phase_eval; phase_export ;;
    -h|--help)    usage ;;
    *)            die "unknown phase: $phase (see --help)" ;;
  esac
done
log "done. results: outputs/eval/metrics-test.json and outputs/eval-router.json"
