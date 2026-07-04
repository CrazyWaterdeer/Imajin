"""Resolve provider *tier* tokens (``sonnet`` / ``opus`` / ``gpt``) to the latest
concrete model id at connection time.

The model picker stores a stable tier for the API-backed Claude and OpenAI
providers instead of a pinned model id, so Imajin tracks new releases without a
code change. Resolution queries each provider's models endpoint, is cached with a
TTL, and falls back to a sane recent default on any error (network down, endpoint
missing, unexpected shape).

The subscription (``claude-agent``) path does **not** use this: its ``sonnet`` /
``opus`` aliases are resolved to the latest model by the Claude CLI itself. Local
models (Ollama) are pinned and never resolved.
"""
from __future__ import annotations

import threading
import time
from typing import Any

# Tier tokens the Anthropic API provider knows how to resolve.
ANTHROPIC_TIERS = ("sonnet", "opus", "haiku")

# Recent, known-good defaults used when live resolution fails. These are the
# floor, not the pin — resolution upgrades past them when the API reports newer.
_ANTHROPIC_FALLBACK = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "haiku": "claude-haiku-4-5",
}
_OPENAI_FALLBACK = {"gpt": "gpt-5"}
# Substrings that mark a non-flagship / non-chat OpenAI model we don't want to
# auto-select as "latest GPT".
_OPENAI_EXCLUDE = (
    "mini",
    "nano",
    "instruct",
    "audio",
    "realtime",
    "image",
    "search",
    "transcribe",
    "tts",
    "embedding",
    "moderation",
    "chat-latest",
    "codex",
)

_OK_TTL = 3600.0  # a successful lookup is good for an hour
_FAIL_TTL = 300.0  # after a failure, retry in 5 minutes rather than every turn
_REQUEST_TIMEOUT = 8.0

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, str]] = {}


def clear_cache() -> None:
    """Drop all cached resolutions (used by tests)."""
    with _lock:
        _cache.clear()


def _cached(key: tuple[str, str], compute, fallback: str) -> str:
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
        if entry is not None and entry[0] > now:
            return entry[1]
    try:
        value = compute()
        ok = bool(value)
    except Exception:  # noqa: BLE001 - resolution must never raise into a turn
        value, ok = None, False
    if not value:
        value = fallback
    with _lock:
        _cache[key] = (time.monotonic() + (_OK_TTL if ok else _FAIL_TTL), value)
    return value


def _pick_newest_anthropic(models: Any, tier: str, fallback: str) -> str:
    cands = [
        m
        for m in models
        if str(getattr(m, "id", "")).startswith("claude-")
        and tier in str(getattr(m, "id", "")).lower()
    ]
    if not cands:
        return fallback
    # created_at is an ISO-8601 string / datetime; str() sorts chronologically.
    return str(max(cands, key=lambda m: str(getattr(m, "created_at", "") or "")).id)


def _pick_newest_openai(models: Any, fallback: str) -> str:
    def is_flagship(model_id: str) -> bool:
        mid = model_id.lower()
        if not mid.startswith("gpt-"):
            return False
        return not any(bad in mid for bad in _OPENAI_EXCLUDE)

    cands = [m for m in models if is_flagship(str(getattr(m, "id", "")))]
    if not cands:
        return fallback
    return str(max(cands, key=lambda m: getattr(m, "created", 0) or 0).id)


def resolve_anthropic_model(client: Any, tier: str) -> str:
    """Latest concrete Claude model id for ``tier`` (``sonnet``/``opus``/``haiku``)."""
    fallback = _ANTHROPIC_FALLBACK.get(tier, tier)

    def compute() -> str:
        models = list(client.with_options(timeout=_REQUEST_TIMEOUT).models.list())
        return _pick_newest_anthropic(models, tier, fallback)

    return _cached(("anthropic", tier), compute, fallback)


def resolve_openai_model(client: Any, family: str = "gpt") -> str:
    """Best-effort latest flagship OpenAI chat model id for ``family``."""
    fallback = _OPENAI_FALLBACK.get(family, "gpt-5")

    def compute() -> str:
        models = list(client.with_options(timeout=_REQUEST_TIMEOUT).models.list())
        return _pick_newest_openai(models, fallback)

    return _cached(("openai", family), compute, fallback)
