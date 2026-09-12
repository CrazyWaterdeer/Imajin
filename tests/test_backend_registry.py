"""Slice 1: the BackendSpec registry is the single source of truth for what
backend kinds exist, how to check them, and how to build them.

These tests are deliberately black-box against imajin.agent.providers.registry's
public surface (BackendSpec, BackendContext, BACKEND_REGISTRY, get_backend,
kinds) rather than against its private per-kind helper functions -- the point
of "single source of truth" is that the TABLE is complete and self-consistent,
not that any one helper has a particular internal shape.
"""
from __future__ import annotations

import pytest

from imajin.agent.providers.registry import (
    BackendContext,
    BackendSpec,
    get_backend,
    kinds,
)

_EXPECTED_KINDS = ("anthropic", "claude-agent", "openai", "codex-agent", "ollama")


def _hermetic_probes(monkeypatch, *, ollama=(False, "Ollama offline"), codex=(False, "codex not found"), claude=(False, "not logged in")):
    """Pin all three dynamic (subprocess/network-backed) probes so registry
    tests never depend on this machine's real Ollama daemon or CLI logins --
    the same probes conftest.py's autouse fixtures pin for ollama/codex-agent
    (claude-agent has no such autouse default, so tests that touch it must
    pin it themselves)."""
    from imajin.ui import provider_status

    monkeypatch.setattr(provider_status, "probe_ollama", lambda *a, **k: ollama)
    monkeypatch.setattr(provider_status, "codex_available", lambda: codex)
    monkeypatch.setattr(provider_status, "subscription_available", lambda: claude)


# ---------------------------------------------------------------------------
# Structural completeness: every kind has both a probe and a factory.
# ---------------------------------------------------------------------------


def test_every_registered_kind_has_a_factory_matching_its_shape():
    """The registry's core promise: no kind is reachable without both a
    factory -- and the factory slot actually used matches
    the declared shape, never both or neither."""
    assert set(kinds()) == set(_EXPECTED_KINDS)
    for kind in kinds():
        spec = get_backend(kind)
        assert spec is not None
        if spec.shape == "provider":
            assert spec.make_provider is not None
            assert spec.make_runner is None
        elif spec.shape == "fused":
            assert spec.make_runner is not None
            assert spec.make_provider is None
        else:
            pytest.fail(f"unknown shape {spec.shape!r} for kind {kind!r}")


def test_get_backend_returns_none_for_an_unregistered_kind():
    """No silent substitution: an unrecognized kind is simply absent, never
    aliased to some other kind's factory (see chat_dock._make_provider /
    _ensure_runner, which turn this None into a RuntimeError rather than
    guessing)."""
    assert get_backend("totally-not-a-real-backend") is None


def test_kinds_preserves_registration_order():
    """cli.py's --doctor prints in this order; pinned so appending a new
    backend can't silently reshuffle the rows already there."""
    assert kinds() == _EXPECTED_KINDS


# ---------------------------------------------------------------------------
# The two shapes stay honest: a "fused" backend never claims a provider
# factory, a "provider" backend never claims a runner factory.
# ---------------------------------------------------------------------------


def test_claude_agent_and_codex_agent_are_the_fused_shape():
    assert get_backend("claude-agent").shape == "fused"
    assert get_backend("codex-agent").shape == "fused"


def test_anthropic_openai_ollama_are_the_provider_shape():
    for kind in ("anthropic", "openai", "ollama"):
        assert get_backend(kind).shape == "provider"


def test_backend_spec_rejects_a_provider_shape_entry_missing_make_provider():
    with pytest.raises(ValueError, match="make_provider"):
        BackendSpec(kind="x", shape="provider", fix_hint=None)


def test_backend_spec_rejects_a_fused_shape_entry_missing_make_runner():
    with pytest.raises(ValueError, match="make_runner"):
        BackendSpec(kind="x", shape="fused", fix_hint=None)


# ---------------------------------------------------------------------------
# Probe correctness: what the registry says about a kind matches what
# provider_status.compute_statuses (the picker's own source of truth) says.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# fix_hint text survived the move from chat_dock._FIX_HINTS verbatim.
# ---------------------------------------------------------------------------


def test_fix_hints_match_the_pre_registry_text():
    assert get_backend("claude-agent").fix_hint == "Run `claude` in a terminal and sign in."
    assert get_backend("codex-agent").fix_hint == "Run `codex login` in a terminal."
    assert (
        get_backend("ollama").fix_hint
        == "Start Ollama, then `ollama pull` a tool-capable model."
    )
    # anthropic/openai have no specific hint -- chat_dock._fix_hint falls back
    # to _DEFAULT_FIX_HINT for these, same as the old _FIX_HINTS.get(kind, default).
    assert get_backend("anthropic").fix_hint is None
    assert get_backend("openai").fix_hint is None


def test_chat_dock_fix_hint_falls_back_to_the_default_for_a_none_hint():
    from imajin.ui.chat_dock import _DEFAULT_FIX_HINT, _fix_hint

    assert _fix_hint("anthropic") == _DEFAULT_FIX_HINT
    assert _fix_hint("openai") == _DEFAULT_FIX_HINT
    assert _fix_hint("totally-not-a-real-backend") == _DEFAULT_FIX_HINT


# ---------------------------------------------------------------------------
# The picker's own hand-maintained rows (_MODEL_CHOICES + live Ollama
# discovery) never name a kind the registry doesn't know about.
# ---------------------------------------------------------------------------


def test_every_static_model_choice_kind_is_registered():
    from imajin.ui.chat_dock import _MODEL_CHOICES

    for _label, kind, _model in _MODEL_CHOICES:
        assert get_backend(kind) is not None, f"{kind!r} has a picker row but no BackendSpec"
    # Discovered/fallback local rows are always kind="ollama" (see
    # chat_dock._build_model_choices) -- not present in the static list above,
    # so check it explicitly.
    assert get_backend("ollama") is not None


# ---------------------------------------------------------------------------
# Requirement: an unknown/unregistered kind is UNAVAILABLE, never usable --
# exercised at the actual dispatch points (ChatDock), not just the registry.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_chat_dock_env(monkeypatch, tmp_path):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [])
    monkeypatch.setattr(
        Settings, "secrets_path", classmethod(lambda cls: tmp_path / "secrets.json")
    )


def _dock_with_mystery_kind_selected(qtbot, viewer):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd
    from imajin.ui.provider_status import ProviderStatus

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    # Simulate the exact footgun the survey flagged: a row reaches the picker
    # for a kind nobody registered a BackendSpec for (a typo, or a picker row
    # added without a matching registry entry). Even marking it "available"
    # must not let it reach a factory.
    dock.model_choices = [("Mystery backend", "mystery-kind", "m1")]
    dock.model_picker = cd._ModelPickerButton(
        dock.model_choices, statuses={"mystery-kind": ProviderStatus(True, None)}
    )
    qtbot.addWidget(dock.model_picker)
    return dock


def test_make_provider_raises_a_clear_error_for_an_unregistered_kind(qtbot, viewer):
    dock = _dock_with_mystery_kind_selected(qtbot, viewer)
    with pytest.raises(RuntimeError, match="mystery-kind"):
        dock._make_provider()


def test_ensure_runner_raises_a_clear_error_for_an_unregistered_kind(qtbot, viewer):
    dock = _dock_with_mystery_kind_selected(qtbot, viewer)
    with pytest.raises(RuntimeError, match="mystery-kind"):
        dock._ensure_runner()


# ---------------------------------------------------------------------------
# Ollama's BackendSpec wires the real factory + wrap functions (not stubs) --
# a light integration check; the domain logic itself (num_ctx sizing, vision,
# loud degradation) is exhaustively covered through ChatDock in
# tests/test_local_model_picker.py and is not re-tested here.
# ---------------------------------------------------------------------------


def test_ollama_backend_spec_factory_and_wrap_round_trip():
    from imajin.agent.local_models import LocalModel
    from imajin.agent.prompts import build_system_prompt
    from imajin.agent.tool_subset import CORE_TOOL_NAMES

    spec = get_backend("ollama")
    warnings: list[str] = []
    model = LocalModel(
        name="qwen3.5:9b",
        context_length=262144,
        capabilities=frozenset({"completion", "tools", "vision"}),
        parameter_size="9.7B",
        size_bytes=6_000_000_000,
    )

    class _Settings:
        ollama_base_url = "http://localhost:11434/v1"

    ctx = BackendContext(
        settings=_Settings(),
        model="qwen3.5:9b",
        local_models={"qwen3.5:9b": model},
        append_system=warnings.append,
    )
    provider = spec.make_provider(ctx)
    assert provider.model == "qwen3.5:9b"
    assert provider.supports_vision is True

    wrapped, prompt = spec.wrap_for_runner(provider)
    assert wrapped._inner is provider
    # The reduced prompt is scoped to exactly the core set, not the full
    # registry -- same call imajin.agent.prompts.build_system_prompt itself
    # is tested against (see test_chat_dock_tool_scoping.py), so this checks
    # the WIRING (wrap_for_runner reaches it with the right tool names), not
    # prompts.py's own reduction logic.
    assert prompt == build_system_prompt(available_tools=set(CORE_TOOL_NAMES))
    assert len(prompt) < len(build_system_prompt())
    assert any("core set" in w for w in warnings)


def test_every_registered_kind_is_covered_by_compute_statuses():
    """The registry owns dispatch; provider_status owns availability. Nothing
    enforces that split across the two modules, so this does.

    A kind registered here but missing from compute_statuses fails in two
    directions at once, both silent: the picker greys it out as "not registered"
    (chat_dock._UNREGISTERED_STATUS) and `imajin doctor` drops the row entirely
    (cli._doctor skips a kind whose status is None). That is the same shape as
    the bug that made an unregistered backend read as AVAILABLE — caught for the
    picker, missed for the doctor — so pin it here rather than rediscovering it
    from a user report.
    """
    from unittest.mock import patch

    from imajin.agent.providers.registry import kinds
    from imajin.config import Settings
    from imajin.ui import provider_status

    settings = Settings(anthropic_api_key="k", openai_api_key="k")
    with (
        patch.object(provider_status, "probe_ollama", return_value=(True, None)),
        patch.object(provider_status, "subscription_available", return_value=(True, None)),
        patch.object(provider_status, "codex_available", return_value=(True, None)),
    ):
        statuses = provider_status.compute_statuses(settings)

    missing = [k for k in kinds() if k not in statuses]
    assert not missing, (
        f"registered but invisible to the picker and to `imajin doctor`: {missing}. "
        "Add a compute_statuses entry, or drop the BackendSpec."
    )
