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
    with patch.object(provider_status, "is_running", return_value=True):
        statuses = provider_status.compute_statuses(s)
    assert statuses["anthropic"].available is True
    assert statuses["openai"].available is True
    assert statuses["ollama"].available is True
    assert statuses["anthropic"].reason is None


def test_anthropic_unavailable_without_key() -> None:
    s = _settings(anthropic_api_key=None)
    with patch.object(provider_status, "is_running", return_value=True):
        statuses = provider_status.compute_statuses(s)
    assert statuses["anthropic"].available is False
    assert statuses["anthropic"].reason == "no API key"


def test_openai_unavailable_without_key() -> None:
    s = _settings(openai_api_key=None)
    with patch.object(provider_status, "is_running", return_value=True):
        statuses = provider_status.compute_statuses(s)
    assert statuses["openai"].available is False


def test_ollama_unavailable_when_offline() -> None:
    s = _settings()
    with patch.object(provider_status, "is_running", return_value=False):
        statuses = provider_status.compute_statuses(s)
    assert statuses["ollama"].available is False
    assert statuses["ollama"].reason == "Ollama offline"


def test_all_unavailable_on_laptop_scenario() -> None:
    # No keys set + Ollama not installed/running.
    s = _settings(anthropic_api_key=None, openai_api_key=None)
    with patch.object(provider_status, "is_running", return_value=False):
        statuses = provider_status.compute_statuses(s)
    assert all(not st.available for st in statuses.values())
