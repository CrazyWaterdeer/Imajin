"""Discover locally-installed Ollama models and size their context window.

Ollama's OpenAI-compatible endpoint (``/v1/chat/completions``) silently ignores
``num_ctx`` no matter how it is passed — top-level, ``extra_body``, or
``extra_body.options`` all still truncate to Ollama's 4096-token default. With
Imajin's real system prompt + 104 tool schemas (~28,300 prompt tokens) that
truncation leaves the model unable to see its own tool list, and it narrates
calling a tool instead of actually emitting a tool call. Only the *native*
``POST /api/chat`` honours ``{"options": {"num_ctx": N}}`` (see
:class:`imajin.agent.providers.ollama.OllamaProvider`), and that endpoint needs
to know, per model, whether it even advertises "tools" support and how large a
context window it can plausibly be asked for. That is what this module
discovers.

Cached with a TTL in the same shape as :mod:`imajin.agent.model_catalog`'s
``_cached`` helper, so a flaky or offline daemon never stalls a turn or the
model picker's status probe — every public function here returns a safe
fallback instead of raising.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

# A successful discovery is good for 5 minutes; a failed one is retried after
# 30s rather than every keystroke in the model picker.
_OK_TTL = 300.0
_FAIL_TTL = 30.0

_NUM_CTX_MULTIPLE = 8192
_NUM_CTX_HEADROOM = 1.5

_lock = threading.Lock()
# Keyed by the normalised (native) base URL. Value is (expires_at, models,
# reachable) — `reachable` is tracked alongside the model list (rather than
# derived from an empty list) because "daemon offline" and "daemon up with zero
# models pulled" both produce an empty list but are different states that
# probe_ollama must tell apart.
_cache: dict[str, tuple[float, list[LocalModel], bool]] = {}


@dataclass(frozen=True)
class LocalModel:
    name: str  # "qwen3.5:9b"
    context_length: int | None  # from model_info["<arch>.context_length"], None if unknown
    capabilities: frozenset[str]  # e.g. frozenset({"completion","tools","vision","thinking"})
    parameter_size: str | None  # "9.7B"
    size_bytes: int | None

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.capabilities


def native_base_url(base_url: str) -> str:
    """Normalise an OpenAI-compat URL to the Ollama native root: strip a trailing
    '/v1' (and any trailing slash). 'http://localhost:11434/v1' -> 'http://localhost:11434'.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed.rstrip("/")


def clear_cache() -> None:
    """Drop all cached discovery results (used by tests)."""
    with _lock:
        _cache.clear()


def _request_json(url: str, timeout: float, payload: dict[str, Any] | None = None) -> Any:
    """GET (payload=None) or POST (payload=dict) `url` and return the parsed body.

    The one network seam in this module — urllib.Request already defaults to
    POST when `data` is given and GET otherwise, so one function covers both
    /api/tags (GET) and /api/show (POST) and tests only need to monkeypatch
    this single call, not urllib internals.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _row_capabilities(row: dict[str, Any]) -> frozenset[str] | None:
    """Capabilities already embedded on an /api/tags row (newer Ollama), else None."""
    caps = row.get("capabilities")
    if isinstance(caps, list) and caps:
        return frozenset(str(c) for c in caps)
    return None


def _context_length_from_model_info(model_info: dict[str, Any]) -> int | None:
    arch = model_info.get("general.architecture")
    if isinstance(arch, str):
        value = model_info.get(f"{arch}.context_length")
        if isinstance(value, int):
            return value
    # general.architecture is missing (or its own context_length key is), so
    # fall back to scanning for any architecture-prefixed context_length key
    # rather than giving up on a value we can plainly see in model_info.
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def _build_model(row: dict[str, Any], show: dict[str, Any] | None) -> LocalModel:
    name = str(row.get("name") or row.get("model") or "")
    details = row.get("details")
    parameter_size = details.get("parameter_size") if isinstance(details, dict) else None
    size_bytes = row.get("size")

    capabilities = _row_capabilities(row)
    context_length: int | None = None
    if capabilities is None:
        # Capabilities weren't on the /api/tags row, so /api/show was consulted
        # for this model — that response is also the only place context length
        # comes from, hence both are derived here together.
        show = show or {}
        raw_caps = show.get("capabilities")
        capabilities = (
            frozenset(str(c) for c in raw_caps) if isinstance(raw_caps, list) else frozenset()
        )
        model_info = show.get("model_info")
        if isinstance(model_info, dict):
            context_length = _context_length_from_model_info(model_info)

    return LocalModel(
        name=name,
        context_length=context_length,
        capabilities=capabilities,
        parameter_size=parameter_size if isinstance(parameter_size, str) else None,
        size_bytes=size_bytes if isinstance(size_bytes, int) else None,
    )


def _discover_uncached(base: str, timeout: float) -> list[LocalModel]:
    tags = _request_json(f"{base}/api/tags", timeout)
    rows = tags.get("models") if isinstance(tags, dict) else None
    if not isinstance(rows, list):
        return []

    models: list[LocalModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("model")
        if not name:
            continue
        show: dict[str, Any] | None = None
        if _row_capabilities(row) is None:
            try:
                candidate = _request_json(f"{base}/api/show", timeout, {"model": name})
            except Exception:  # noqa: BLE001 - one model's /api/show failing must not hide the rest
                candidate = None
            show = candidate if isinstance(candidate, dict) else None
        models.append(_build_model(row, show))
    return models


def _discover_with_status(base: str, timeout: float, force: bool) -> tuple[list[LocalModel], bool]:
    """(models, reachable), cached. Shared by discover_ollama_models (which only
    needs the list) and probe_ollama (which needs "offline" told apart from
    "online but empty" — a bare model list can't carry that distinction, so the
    reachability bit is cached alongside it instead of re-derived from it).
    """
    now = time.monotonic()
    if not force:
        with _lock:
            entry = _cache.get(base)
            if entry is not None and entry[0] > now:
                return entry[1], entry[2]
    try:
        models = _discover_uncached(base, timeout)
        reachable = True
    except Exception:  # noqa: BLE001 - discovery must never raise into a turn or a UI probe
        models = []
        reachable = False
    with _lock:
        ttl = _OK_TTL if reachable else _FAIL_TTL
        _cache[base] = (time.monotonic() + ttl, models, reachable)
    return models, reachable


def discover_ollama_models(
    base_url: str, *, timeout: float = 3.0, force: bool = False
) -> list[LocalModel]:
    """GET /api/tags then POST /api/show per model for capabilities + context length.
    Cached with a TTL (success 300s, failure 30s). Returns [] on any error — never
    raises. force=True bypasses the cache.
    """
    models, _reachable = _discover_with_status(native_base_url(base_url), timeout, force)
    return models


def probe_ollama(base_url: str, *, timeout: float = 1.5) -> tuple[bool, str | None]:
    """(available, reason_if_not). Distinguishes: daemon down -> 'Ollama offline';
    daemon up but zero models -> 'no models pulled'; daemon up but no tool-capable
    model -> 'no tool-capable model'. Returns (True, None) when usable. Never raises.
    """
    models, reachable = _discover_with_status(native_base_url(base_url), timeout, force=False)
    if not reachable:
        return False, "Ollama offline"
    if not models:
        return False, "no models pulled"
    if not any(m.supports_tools for m in models):
        return False, "no tool-capable model"
    return True, None


def choose_num_ctx(model: LocalModel, estimated_prompt_tokens: int, *, floor: int = 32768) -> int:
    """Context window to request. Round (estimated * 1.5) up to the next multiple of 8192,
    take max with floor, then clamp to model.context_length when known. When
    model.context_length is known and smaller than the estimate, return the model's
    max (the caller warns) rather than an impossible value.
    """
    target = estimated_prompt_tokens * _NUM_CTX_HEADROOM
    rounded = math.ceil(target / _NUM_CTX_MULTIPLE) * _NUM_CTX_MULTIPLE
    value = max(rounded, floor)
    if model.context_length is not None:
        # Also covers "estimate exceeds the model's real max": min() already
        # clamps down to model.context_length in that case, which *is* the
        # model's max — there is nothing larger we could honestly request.
        value = min(value, model.context_length)
    return value


def _safe_json_len(payload: Any) -> int:
    try:
        return len(json.dumps(payload, default=str))
    except TypeError:
        return len(str(payload))


def estimate_prompt_tokens(
    system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> int:
    """Cheap char/3.5 heuristic over the serialized payload. Deliberately approximate."""
    chars = len(system) + _safe_json_len(tools) + _safe_json_len(messages)
    return int(chars / 3.5)
