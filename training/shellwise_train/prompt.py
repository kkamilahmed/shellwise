"""Single source of truth for the prompt format.

The Rust engine must reproduce this byte-for-byte, so keep it minimal and
document any change in the top-level README.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You translate natural language into a single POSIX shell command. Reply with only the command."
)


def build_messages(nl: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl.strip()},
    ]


def render_prompt(tokenizer, nl: str) -> str:
    """Chat-templated prompt ending right where the assistant's reply starts."""
    return tokenizer.apply_chat_template(
        build_messages(nl), tokenize=False, add_generation_prompt=True
    )


def render_completion(tokenizer, cmd: str) -> str:
    """The target the model should emit: the command followed by end-of-turn."""
    return cmd.strip() + tokenizer.eos_token
