from __future__ import annotations

from unittest.mock import patch

from imajin.config import Settings
from imajin.ui import provider_status


def _settings(**overrides) -> Settings:
    base = dict(
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-oai-test",
        ollama_base_url="http://localhost:11434/v1",
    )
    base.update(overrides)
    return Settings(**base)


def test_all_available_when_keys_set_and_ollama_up() -> None:
    s = _settings()
    with patch.object(provider_status, "probe_ollama", return_value=(True, None)):
        statuses = provider_status.compute_statuses(s)
    assert statuses["anthropic"].available is True
    assert statuses["openai"].available is True
    assert statuses["ollama"].available is True
    assert statuses["anthropic"].reason is None


def test_anthropic_unavailable_without_key() -> None:
    s = _settings(anthropic_api_key=None)
    with patch.object(provider_status, "probe_ollama", return_value=(True, None)):
        statuses = provider_status.compute_statuses(s)
    assert statuses["anthropic"].available is False
    assert statuses["anthropic"].reason == "no API key"


def test_openai_unavailable_without_key() -> None:
    s = _settings(openai_api_key=None)
    with patch.object(provider_status, "probe_ollama", return_value=(True, None)):
        statuses = provider_status.compute_statuses(s)
    assert statuses["openai"].available is False


def test_ollama_unavailable_when_offline() -> None:
    s = _settings()
    with patch.object(provider_status, "probe_ollama", return_value=(False, "Ollama offline")):
        statuses = provider_status.compute_statuses(s)
    assert statuses["ollama"].available is False
    assert statuses["ollama"].reason == "Ollama offline"


def test_ollama_unavailable_when_no_models_pulled() -> None:
    # Daemon is up (TCP connects) but `ollama pull` was never run -- distinct
    # from "offline" so the picker can tell the user what to actually do.
    s = _settings()
    with patch.object(
        provider_status, "probe_ollama", return_value=(False, "no models pulled")
    ):
        statuses = provider_status.compute_statuses(s)
    assert statuses["ollama"].available is False
    assert statuses["ollama"].reason == "no models pulled"


def test_ollama_unavailable_when_no_tool_capable_model() -> None:
    # Daemon up, models pulled, but none advertise "tools" -- Imajin's agent
    # loop is unusable without tool calling, so this must read as unavailable
    # rather than a green light that fails on the first turn.
    s = _settings()
    with patch.object(
        provider_status, "probe_ollama", return_value=(False, "no tool-capable model")
    ):
        statuses = provider_status.compute_statuses(s)
    assert statuses["ollama"].available is False
    assert statuses["ollama"].reason == "no tool-capable model"


def test_all_unavailable_on_laptop_scenario() -> None:
    # No keys set + Ollama not installed/running + no `claude`/`codex` login.
    s = _settings(anthropic_api_key=None, openai_api_key=None)
    with (
        patch.object(provider_status, "probe_ollama", return_value=(False, "Ollama offline")),
        patch.object(
            provider_status, "subscription_available", return_value=(False, "not logged in")
        ),
        patch.object(
            provider_status, "codex_available", return_value=(False, "codex not found")
        ),
    ):
        statuses = provider_status.compute_statuses(s)
    assert all(not st.available for st in statuses.values())


def test_subscription_available_without_api_keys() -> None:
    # The subscription agent is independent of API keys: a logged-in `claude`
    # makes it available even when no keys are configured.
    s = _settings(anthropic_api_key=None, openai_api_key=None)
    with (
        patch.object(provider_status, "probe_ollama", return_value=(False, "Ollama offline")),
        patch.object(provider_status, "subscription_available", return_value=(True, None)),
    ):
        statuses = provider_status.compute_statuses(s)
    assert statuses["claude-agent"].available is True
    assert statuses["anthropic"].available is False


def test_codex_agent_available_without_api_keys() -> None:
    # Same independence as claude-agent above -- codex_available() checks a
    # `codex` CLI login, not an API key, so it can be True with no keys set.
    s = _settings(anthropic_api_key=None, openai_api_key=None)
    with (
        patch.object(provider_status, "probe_ollama", return_value=(False, "Ollama offline")),
        patch.object(provider_status, "codex_available", return_value=(True, None)),
    ):
        statuses = provider_status.compute_statuses(s)
    assert statuses["codex-agent"].available is True
    assert statuses["anthropic"].available is False


def test_codex_agent_unavailable_reason_flows_through() -> None:
    # Mirrors test_ollama_unavailable_when_no_models_pulled's intent for the
    # codex-agent kind: compute_statuses must pass codex_available()'s own
    # reason string through unchanged, not collapse it to a generic label.
    s = _settings()
    with (
        patch.object(provider_status, "probe_ollama", return_value=(True, None)),
        patch.object(provider_status, "codex_available", return_value=(False, "not logged in")),
    ):
        statuses = provider_status.compute_statuses(s)
    assert statuses["codex-agent"].available is False
    assert statuses["codex-agent"].reason == "not logged in"
