"""Single source of truth for the prompt format.

The Rust engine must reproduce this byte-for-byte, so keep it minimal and
document any change in the top-level README.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You translate natural language into a single POSIX shell command. Reply with only the command."
)


def build_user_turn(nl: str, tool: str | None = None) -> str:
    """The user message: an optional ``tool:`` line from the router, then the request.

    The ``request:`` prefix is always present so the layout is identical with and
    without a hint; only the tool line comes and goes.
    """
    lines = []
    if tool:
        lines.append(f"tool: {tool.strip()}")
    lines.append(f"request: {nl.strip()}")
    return "\n".join(lines)


def build_messages(nl: str, tool: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(nl, tool)},
    ]


def render_prompt(tokenizer, nl: str, tool: str | None = None) -> str:
    """Chat-templated prompt ending right where the assistant's reply starts."""
    return tokenizer.apply_chat_template(
        build_messages(nl, tool), tokenize=False, add_generation_prompt=True
    )


def render_completion(tokenizer, cmd: str) -> str:
    """The target the model should emit: the command followed by end-of-turn."""
    return cmd.strip() + tokenizer.eos_token
