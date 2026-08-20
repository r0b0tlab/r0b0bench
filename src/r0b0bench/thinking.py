from __future__ import annotations

import json
import os


def thinking_on() -> bool:
    raw = os.environ.get("R0B0BENCH_CHAT_TEMPLATE_KWARGS")
    if not raw:
        return False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and (
        value.get("enable_thinking") is True or value.get("thinking") is True
    )


def effective_max_tokens(default: int, lane: str) -> int:
    """Use generous completion budgets when explicit reasoning is enabled."""
    if not thinking_on():
        return default
    override = os.environ.get("R0B0BENCH_THINKING_MAX_TOKENS")
    if override:
        return max(default, int(override))
    lane_defaults = {
        "gsm8k": 49_152,
        "qa": 32_768,
        "ifeval": 32_768,
        "humaneval": 32_768,
    }
    return max(default, lane_defaults.get(lane, 8_192))


def effective_generation_reserve(default: int) -> int:
    """Reserve enough generation room for explicit reasoning in NIAH."""
    if not thinking_on():
        return default
    override = os.environ.get("R0B0BENCH_THINKING_GENERATION_RESERVE")
    return max(default, int(override)) if override else max(default, 8_192)
