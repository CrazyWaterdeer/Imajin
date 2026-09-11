"""Tests for Ollama model discovery, capability/context parsing, and num_ctx sizing.

The one network seam (`_request_json`) is faked throughout, so nothing here needs
a real Ollama daemon or network access. The fake /api/tags and /api/show payloads
mirror the real qwen3.5:9b shapes this module was built against (see the task's
measured-ground-truth notes: qwen35.context_length = 262144).
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from imajin.agent import local_models as lm

BASE = "http://localhost:11434/v1"


@pytest.fixture(autouse=True)
def _clear_cache():
    lm.clear_cache()
    yield
    lm.clear_cache()


# Real /api/tags row shape (0.22.0): no "capabilities" field, so discovery must
# fall back to /api/show for capabilities *and* context length.
_QWEN_TAGS_ROW: dict[str, Any] = {
    "name": "qwen3.5:9b",
    "model": "qwen3.5:9b",
    "modified_at": "2026-01-01T00:00:00Z",
    "size": 6_500_000_000,
    "digest": "sha256:deadbeef",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen35",
        "families": ["qwen35"],
        "parameter_size": "9.7B",
        "quantization_level": "Q4_K_M",
    },
}

_QWEN_SHOW: dict[str, Any] = {
    "details": _QWEN_TAGS_ROW["details"],
    "model_info": {
        "general.architecture": "qwen35",
        "general.parameter_count": 9_700_000_000,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 4096,
    },
    "capabilities": ["completion", "tools", "vision", "thinking"],
}


class _FakeTransport:
    """Stands in for local_models._request_json. Routes by URL suffix and counts
    calls so tests can assert directly on caching (no second request issued)."""

    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        shows: dict[str, dict[str, Any]] | None = None,
        raise_on_tags: Exception | None = None,
        raise_on_show: Exception | None = None,
    ) -> None:
        self.tags = tags if tags is not None else {"models": [_QWEN_TAGS_ROW]}
        self.shows = shows if shows is not None else {"qwen3.5:9b": _QWEN_SHOW}
        self.raise_on_tags = raise_on_tags
        self.raise_on_show = raise_on_show
        self.tags_calls = 0
        self.show_calls = 0

    def __call__(self, url: str, timeout: float, payload: dict[str, Any] | None = None) -> Any:
        if url.endswith("/api/tags"):
            self.tags_calls += 1
            if self.raise_on_tags is not None:
                raise self.raise_on_tags
            return self.tags
        if url.endswith("/api/show"):
            self.show_calls += 1
            if self.raise_on_show is not None:
                raise self.raise_on_show
            assert payload is not None
            return self.shows[payload["model"]]
        raise AssertionError(f"unexpected URL: {url}")


# -- native_base_url ----------------------------------------------------------


def test_native_base_url_strips_v1() -> None:
    assert lm.native_base_url("http://localhost:11434/v1") == "http://localhost:11434"


def test_native_base_url_no_v1_is_unchanged() -> None:
    assert lm.native_base_url("http://localhost:11434") == "http://localhost:11434"


def test_native_base_url_strips_trailing_slash() -> None:
    assert lm.native_base_url("http://localhost:11434/") == "http://localhost:11434"
    assert lm.native_base_url("http://localhost:11434/v1/") == "http://localhost:11434"


# -- discovery: capability + context-length parsing ----------------------------


def test_discover_parses_qwen35_context_length_and_capabilities(monkeypatch) -> None:
    fake = _FakeTransport()
    monkeypatch.setattr(lm, "_request_json", fake)

    models = lm.discover_ollama_models(BASE)

    assert len(models) == 1
    model = models[0]
    assert model.name == "qwen3.5:9b"
    assert model.context_length == 262144
    assert model.capabilities == frozenset({"completion", "tools", "vision", "thinking"})
    assert model.parameter_size == "9.7B"
    assert model.size_bytes == 6_500_000_000
    assert model.supports_tools is True
    assert model.supports_vision is True
    assert fake.tags_calls == 1
    assert fake.show_calls == 1


def test_discover_uses_row_capabilities_and_skips_show(monkeypatch) -> None:
    row = dict(_QWEN_TAGS_ROW, capabilities=["completion", "tools"])
    fake = _FakeTransport(tags={"models": [row]})
    monkeypatch.setattr(lm, "_request_json", fake)

    models = lm.discover_ollama_models(BASE)

    assert len(models) == 1
    assert models[0].capabilities == frozenset({"completion", "tools"})
    # Context length only ever comes from /api/show; skipping that call (because
    # the tags row already had capabilities) means it stays unknown, not guessed.
    assert models[0].context_length is None
    assert fake.tags_calls == 1
    assert fake.show_calls == 0


def test_context_length_falls_back_when_architecture_key_absent(monkeypatch) -> None:
    show = {
        "model_info": {"weirdarch.context_length": 8192},
        "capabilities": ["completion", "tools"],
    }
    fake = _FakeTransport(shows={"qwen3.5:9b": show})
    monkeypatch.setattr(lm, "_request_json", fake)

    models = lm.discover_ollama_models(BASE)
    assert models[0].context_length == 8192


# -- caching --------------------------------------------------------------


def test_discovery_is_cached_second_call_makes_no_request(monkeypatch) -> None:
    fake = _FakeTransport()
    monkeypatch.setattr(lm, "_request_json", fake)

    first = lm.discover_ollama_models(BASE)
    second = lm.discover_ollama_models(BASE)

    assert first == second
    assert fake.tags_calls == 1
    assert fake.show_calls == 1


def test_force_bypasses_cache(monkeypatch) -> None:
    fake = _FakeTransport()
    monkeypatch.setattr(lm, "_request_json", fake)

    lm.discover_ollama_models(BASE)
    lm.discover_ollama_models(BASE, force=True)

    assert fake.tags_calls == 2


def test_discover_returns_empty_list_on_connection_error(monkeypatch) -> None:
    fake = _FakeTransport(raise_on_tags=ConnectionRefusedError("refused"))
    monkeypatch.setattr(lm, "_request_json", fake)

    assert lm.discover_ollama_models(BASE) == []


# -- probe_ollama -----------------------------------------------------------


def test_probe_offline_when_daemon_unreachable(monkeypatch) -> None:
    fake = _FakeTransport(raise_on_tags=ConnectionRefusedError("refused"))
    monkeypatch.setattr(lm, "_request_json", fake)

    assert lm.probe_ollama(BASE) == (False, "Ollama offline")


def test_probe_no_models_pulled(monkeypatch) -> None:
    fake = _FakeTransport(tags={"models": []})
    monkeypatch.setattr(lm, "_request_json", fake)

    assert lm.probe_ollama(BASE) == (False, "no models pulled")


def test_probe_no_tool_capable_model(monkeypatch) -> None:
    show = {
        "model_info": {"general.architecture": "qwen35", "qwen35.context_length": 262144},
        "capabilities": ["completion", "vision"],  # no "tools"
    }
    fake = _FakeTransport(shows={"qwen3.5:9b": show})
    monkeypatch.setattr(lm, "_request_json", fake)

    assert lm.probe_ollama(BASE) == (False, "no tool-capable model")


def test_probe_ok_when_tool_capable_model_present(monkeypatch) -> None:
    fake = _FakeTransport()
    monkeypatch.setattr(lm, "_request_json", fake)

    assert lm.probe_ollama(BASE) == (True, None)


# -- choose_num_ctx -----------------------------------------------------------

_UNBOUNDED = lm.LocalModel(
    name="unbounded",
    context_length=None,
    capabilities=frozenset(),
    parameter_size=None,
    size_bytes=None,
)


def test_choose_num_ctx_rounds_up_to_multiple_of_8192() -> None:
    # 10_000 * 1.5 = 15_000 -> next multiple of 8192 is 16384.
    assert lm.choose_num_ctx(_UNBOUNDED, 10_000, floor=0) == 16384


def test_choose_num_ctx_respects_floor() -> None:
    assert lm.choose_num_ctx(_UNBOUNDED, 100, floor=32768) == 32768


def test_choose_num_ctx_clamps_to_model_max() -> None:
    model = lm.LocalModel(
        name="m", context_length=40_000, capabilities=frozenset(), parameter_size=None, size_bytes=None
    )
    # Uncapped, 100_000 * 1.5 rounds to 155_648 -- well above the model's max.
    assert lm.choose_num_ctx(model, 100_000, floor=0) == 40_000


def test_choose_num_ctx_returns_model_max_when_estimate_exceeds_it() -> None:
    model = lm.LocalModel(
        name="tiny", context_length=4096, capabilities=frozenset(), parameter_size=None, size_bytes=None
    )
    assert lm.choose_num_ctx(model, 28_320) == 4096


# -- estimate_prompt_tokens ----------------------------------------------------


def test_estimate_prompt_tokens_matches_char_over_3_5_heuristic() -> None:
    system = "x" * 350
    tools = [{"name": "register_files", "input_schema": {"type": "object"}}]
    messages = [{"role": "user", "content": "hi"}]

    expected_chars = len(system) + len(json.dumps(tools)) + len(json.dumps(messages))
    assert lm.estimate_prompt_tokens(system, tools, messages) == int(expected_chars / 3.5)


def test_estimate_prompt_tokens_empty_inputs() -> None:
    assert lm.estimate_prompt_tokens("", [], []) == int(len("[]") * 2 / 3.5)
