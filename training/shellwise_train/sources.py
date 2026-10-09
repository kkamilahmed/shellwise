"""Public NL-to-shell datasets and loaders that map them to a common ``Pair``.

Every loader yields raw pairs. Filtering, normalisation and dedup happen in
``build_dataset`` so the rules live in one place.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .normalize import clean_completion, strip_placeholders, strip_trailing_output


@dataclass(frozen=True)
class Pair:
    nl: str
    cmd: str
    source: str


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    files: tuple[str, ...]
    license: str
    load: Callable[[dict[str, Path]], Iterator[Pair]]


def _load_nl2bash_parquet(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        df = pd.read_parquet(path)
        for nl, cmd in zip(df["nl"], df["bash"], strict=True):
            yield Pair(str(nl), str(cmd), "nl2bash")


def _load_nl2bash_custom(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            yield Pair(str(row["nl_command"]), str(row["bash_code"]), "nl2bash-custom")


def _load_tldr(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        df = pd.read_json(path, lines=True)
        for nl, cmd in zip(df["nl"], df["cmd"], strict=True):
            yield Pair(str(nl), strip_placeholders(str(cmd)), "tldr")


def _load_prompt_response(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            yield Pair(str(row["prompt"]), str(row["response"]), "sysadmin-840")


_SCRIPT_PROMPT_RE = re.compile(r"^\s*write a (bash|shell) script", re.IGNORECASE)


def _load_bash6k(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        df = pd.read_parquet(path)
        for prompt, completion in zip(df["prompt"], df["completion"], strict=True):
            if _SCRIPT_PROMPT_RE.match(str(prompt)):
                continue
            cmd = clean_completion(str(completion))
            if cmd is None or cmd.startswith("#!"):
                continue
            yield Pair(str(prompt), strip_trailing_output(cmd), "bash-6k")


_JAWA_PREFIX = "Write the Linux bash command for this task:"


def parse_jawa_messages(raw: str) -> tuple[str, str] | None:
    """The ``messages`` column is a Python-repr list, not JSON."""
    try:
        messages = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    user = next((m["content"] for m in messages if m.get("role") == "user"), None)
    assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
    if not user or not assistant:
        return None
    if user.startswith(_JAWA_PREFIX):
        user = user[len(_JAWA_PREFIX) :]
    return user.strip(), assistant.strip()


def _load_jawa(files: dict[str, Path]) -> Iterator[Pair]:
    for path in files.values():
        df = pd.read_json(path, lines=True)
        for raw in df["messages"]:
            parsed = parse_jawa_messages(str(raw))
            if parsed:
                yield Pair(parsed[0], parsed[1], "linux-bash-sft")


SOURCES: tuple[Source, ...] = (
    Source(
        name="nl2bash",
        repo="GWHed/nl2bash",
        files=(
            "data/train-00000-of-00001.parquet",
            "data/dev-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ),
        license="GPL-3.0 (Tellina NL2Bash corpus)",
        load=_load_nl2bash_parquet,
    ),
    Source(
        name="nl2bash-custom",
        repo="AnishJoshi/nl2bash-custom",
        files=("data/train.json", "data/dev.json", "data/test.json"),
        license="unspecified (NL2Bash derivative)",
        load=_load_nl2bash_custom,
    ),
    Source(
        name="tldr",
        repo="neulab/tldr",
        files=("tldr-train.jsonl", "tldr-dev.jsonl", "tldr-test.jsonl"),
        license="MIT",
        load=_load_tldr,
    ),
    Source(
        name="sysadmin-840",
        repo="aelhalili/bash-commands-dataset",
        files=("dataset.json",),
        license="MIT",
        load=_load_prompt_response,
    ),
    Source(
        name="bash-6k",
        repo="emirkaanozdemr/bash_command_data_6K",
        files=("data/train-00000-of-00001.parquet",),
        license="Apache-2.0",
        load=_load_bash6k,
    ),
    Source(
        name="linux-bash-sft",
        repo="Jawajawa/command-linux-bash-balanced-sft",
        files=("train.jsonl",),
        license="unspecified (NL2Bash derivative)",
        load=_load_jawa,
    ),
)


def download(source: Source, cache_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        f: Path(
            hf_hub_download(
                repo_id=source.repo, filename=f, repo_type="dataset", cache_dir=str(cache_dir)
            )
        )
        for f in source.files
    }
