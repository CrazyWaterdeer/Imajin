"""Compute availability of each LLM provider for the model picker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imajin.agent.local_models import probe_ollama
from imajin.agent.providers.claude_agent import subscription_available


@dataclass(frozen=True)
class ProviderStatus:
    available: bool
    reason: str | None  # short, shown next to the menu label when unavailable


_OK = ProviderStatus(available=True, reason=None)


def compute_statuses(settings: Any) -> dict[str, ProviderStatus]:
    """Return availability for each provider kind used in the picker."""
    statuses: dict[str, ProviderStatus] = {}

    statuses["anthropic"] = (
        _OK
        if settings.anthropic_api_key
        else ProviderStatus(available=False, reason="no API key")
    )
    # Subscription-backed agent needs no API key — just a logged-in `claude` CLI.
    sub_ok, sub_reason = subscription_available()
    statuses["claude-agent"] = (
        _OK if sub_ok else ProviderStatus(available=False, reason=sub_reason)
    )
    statuses["openai"] = (
        _OK
        if settings.openai_api_key
        else ProviderStatus(available=False, reason="no API key")
    )
    # probe_ollama (not the bare TCP check in ollama_helper.is_running) so a
    # daemon that's up but has nothing pulled, or nothing tool-capable, shows
    # its real reason in the picker instead of a green light that only fails
    # once the user actually sends a message.
    ollama_ok, ollama_reason = probe_ollama(settings.ollama_base_url, timeout=0.5)
    statuses["ollama"] = (
        _OK if ollama_ok else ProviderStatus(available=False, reason=ollama_reason)
    )

    return statuses
