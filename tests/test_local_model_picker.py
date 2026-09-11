"""Slice D: dynamic local-model picker, native Ollama provider wiring, and the
loud-degradation warning when a model's context can't hold the full tool set.

tests/test_chat_dock_phase3.py keeps the pre-existing picker-shape assertions
(now discovery-agnostic); this file covers the behavior that's new in this
slice specifically.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_chat_dock_env(monkeypatch, tmp_path):
    """Every test here constructs a ChatDock, which has two real-world side
    effects unless redirected:
      - ChatDock.__init__ calls discover_ollama_models unconditionally (see
        chat_dock._build_model_choices) -- default that to "found nothing" so
        no test in this file touches a real Ollama daemon.
      - clicking the picker persists via Settings.save_secrets() -- point that
        at a throwaway file instead of the shared path conftest.py redirects
        XDG_CONFIG_HOME to, so tests can't see each other's writes.
    """
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [])
    monkeypatch.setattr(
        Settings, "secrets_path", classmethod(lambda cls: tmp_path / "secrets.json")
    )


def _model(
    name: str = "qwen3.5:9b",
    context_length: int | None = 262144,
    capabilities: frozenset[str] = frozenset({"completion", "tools", "vision", "thinking"}),
    parameter_size: str | None = "9.7B",
    size_bytes: int | None = 6_000_000_000,
):
    from imajin.agent.local_models import LocalModel

    return LocalModel(
        name=name,
        context_length=context_length,
        capabilities=capabilities,
        parameter_size=parameter_size,
        size_bytes=size_bytes,
    )


def _all_available():
    from imajin.ui.provider_status import ProviderStatus

    return {
        k: ProviderStatus(True, None) for k in ("anthropic", "claude-agent", "openai", "ollama")
    }


def _select_first_ollama_row(dock) -> None:
    idx = next(i for i, (_, k, _m) in enumerate(dock.model_choices) if k == "ollama")
    dock.model_picker.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def test_local_model_label_includes_parameter_size_and_context() -> None:
    from imajin.ui.chat_dock import _local_model_label

    assert _local_model_label(_model()) == "Local: qwen3.5:9b (9.7B, 256K)"


def test_local_model_label_omits_missing_fields() -> None:
    from imajin.ui.chat_dock import _local_model_label

    m = _model(parameter_size=None, context_length=None)
    assert _local_model_label(m) == "Local: qwen3.5:9b"


def test_local_model_label_handles_parameter_size_only() -> None:
    from imajin.ui.chat_dock import _local_model_label

    m = _model(context_length=None)
    assert _local_model_label(m) == "Local: qwen3.5:9b (9.7B)"


def test_format_context_length_matches_ollama_style() -> None:
    from imajin.ui.chat_dock import _format_context_length

    assert _format_context_length(262144) == "256K"


# ---------------------------------------------------------------------------
# Fallback row
# ---------------------------------------------------------------------------


def test_fallback_local_choice_uses_configured_model() -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import _fallback_local_choice

    row = _fallback_local_choice(Settings(ollama_model="phi4:14b"))
    assert row == ("Local: phi4:14b (unconfirmed)", "ollama", "phi4:14b")


def test_fallback_local_choice_with_nothing_configured() -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import _fallback_local_choice

    row = _fallback_local_choice(Settings())
    assert row == ("Local: none configured", "ollama", "")


# ---------------------------------------------------------------------------
# _build_model_choices / picker rows
# ---------------------------------------------------------------------------


def test_build_model_choices_appends_one_row_per_discovered_model(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    models = [
        _model(name="qwen3.5:9b"),
        _model(name="llama3.1:8b", parameter_size="8.0B", context_length=131072),
    ]
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: models)

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    assert dock.model_picker.count() == len(cd._MODEL_CHOICES) + 2
    labels = [dock.model_picker.itemText(i) for i in range(dock.model_picker.count())]
    assert any("qwen3.5:9b" in text for text in labels)
    assert any("llama3.1:8b" in text and "128K" in text for text in labels)


def test_build_model_choices_filters_out_non_tool_capable_models(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    # A vision-only model with no "tools" capability must not appear -- Imajin
    # always needs tool calling, so surfacing it would just be a picker entry
    # that fails on the first turn.
    no_tools = _model(name="llava:7b", capabilities=frozenset({"completion", "vision"}))
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [no_tools])

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    labels = [dock.model_picker.itemText(i) for i in range(dock.model_picker.count())]
    assert not any("llava:7b" in text for text in labels)
    # Filtered down to nothing tool-capable -> the fallback row, not zero rows.
    assert dock.model_picker.count() == len(cd._MODEL_CHOICES) + 1


def test_build_model_choices_falls_back_to_settings_ollama_model(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    dock = cd.ChatDock(viewer=viewer, settings=Settings(ollama_model="phi4:14b"))
    qtbot.addWidget(dock)

    labels = [dock.model_picker.itemText(i) for i in range(dock.model_picker.count())]
    assert any("phi4:14b" in text for text in labels)


def test_build_model_choices_fallback_present_with_nothing_configured(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    dock = cd.ChatDock(viewer=viewer, settings=Settings())  # ollama_model == ""
    qtbot.addWidget(dock)

    # Never left with zero local rows, even with nothing pulled and nothing
    # configured -- an empty picker looks broken.
    assert dock.model_picker.count() == len(cd._MODEL_CHOICES) + 1


# ---------------------------------------------------------------------------
# invalidate_runner rebuild
# ---------------------------------------------------------------------------


def test_invalidate_runner_picks_up_newly_pulled_model(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())
    discovered = [_model(name="qwen3.5:9b")]
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: discovered)

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    before = dock.model_picker.count()

    discovered.append(_model(name="new-model:latest", parameter_size="3B", context_length=8192))
    dock.invalidate_runner()

    assert dock.model_picker.count() == before + 1
    labels = [dock.model_picker.itemText(i) for i in range(dock.model_picker.count())]
    assert any("new-model:latest" in text for text in labels)


def test_invalidate_runner_keeps_current_selection_when_still_present(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())
    discovered = [_model(name="qwen3.5:9b"), _model(name="llama3.1:8b")]
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: discovered)

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    idx = next(
        i
        for i, (_, k, m) in enumerate(dock.model_choices)
        if k == "ollama" and m == "llama3.1:8b"
    )
    dock.model_picker.setCurrentIndex(idx)

    discovered.append(_model(name="a-third-model:1b"))
    dock.invalidate_runner()

    _, kind, model = dock.model_choices[dock.model_picker.currentIndex()]
    assert (kind, model) == ("ollama", "llama3.1:8b")


# ---------------------------------------------------------------------------
# _make_provider: native Ollama, num_ctx sizing, vision, loud degradation
# ---------------------------------------------------------------------------


def test_make_provider_ollama_uses_native_provider_with_sized_num_ctx(qtbot, viewer, monkeypatch):
    from imajin.agent.providers.ollama import OllamaProvider
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    model = _model(context_length=262144)
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [model])
    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    _select_first_ollama_row(dock)

    provider = dock._make_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3.5:9b"
    # choose_num_ctx never returns more than the model's real context_length.
    assert provider.num_ctx is not None
    assert provider.num_ctx <= 262144
    assert provider.supports_vision is True  # model advertises "vision"


def test_make_provider_ollama_disables_vision_for_text_only_model(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    model = _model(capabilities=frozenset({"completion", "tools"}))
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [model])
    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    _select_first_ollama_row(dock)

    provider = dock._make_provider()

    assert provider.supports_vision is False


def test_make_provider_warns_loudly_when_context_too_small_for_tool_set(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    # Real system-prompt + 104-tool payloads are tens of thousands of tokens
    # (see local_models.py's module docstring) -- a 2048-token model can never
    # hold that, so choose_num_ctx must clamp below the estimate here.
    tiny = _model(name="tiny:1b", context_length=2048, parameter_size="1B")
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [tiny])
    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    _select_first_ollama_row(dock)

    provider = dock._make_provider()

    assert provider.num_ctx == 2048  # clamped to the model's real max
    text = dock.transcript.toPlainText()
    assert "tiny:1b" in text
    assert "2048" in text
    assert "truncated" in text.lower()


def test_make_provider_does_not_warn_when_context_covers_the_prompt(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    model = _model(context_length=262144)
    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [model])
    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    _select_first_ollama_row(dock)

    dock._make_provider()

    text = dock.transcript.toPlainText()
    assert "truncated" not in text.lower()


def test_make_provider_fallback_row_has_no_clamp_and_no_vision(qtbot, viewer, monkeypatch):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [])
    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())

    dock = cd.ChatDock(viewer=viewer, settings=Settings(ollama_model="phi4:14b"))
    qtbot.addWidget(dock)
    _select_first_ollama_row(dock)

    provider = dock._make_provider()

    assert provider.model == "phi4:14b"
    # No known context_length to clamp to -> the headroom-adjusted estimate,
    # rounded up to a multiple of 8192, floored at 16384 (not the old 32768:
    # _make_provider's ollama branch now sizes num_ctx from the 20-tool core
    # subset and passes floor=16384 at the call site, so a legitimately
    # smaller estimate is no longer forced back up to the old default -- see
    # imajin.agent.tool_subset).
    assert provider.num_ctx >= 16384
    assert provider.num_ctx % 8192 == 0
    assert provider.supports_vision is False  # capabilities unknown -> default off


# ---------------------------------------------------------------------------
# provider_status integration: probe_ollama's reason reaches the picker menu
# ---------------------------------------------------------------------------


def test_ollama_status_reason_flows_through_to_the_picker_menu(qtbot, viewer, monkeypatch):
    """End-to-end check of task #4's wiring: provider_status.compute_statuses
    (unmocked here) calls the (mocked) probe_ollama, and that specific reason
    reaches the rendered menu for the "ollama" kind -- not just a bare
    True/False from a TCP connect.
    """
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd
    from imajin.ui import provider_status

    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [])
    monkeypatch.setattr(
        provider_status, "probe_ollama", lambda *a, **k: (False, "no tool-capable model")
    )

    dock = cd.ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    menu_texts = [a.text() for a in dock.model_picker.menu().actions() if not a.isSeparator()]
    assert any("no tool-capable model" in t for t in menu_texts)
