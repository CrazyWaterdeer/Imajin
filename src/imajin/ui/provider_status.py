"""Compute availability of each LLM provider for the model picker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imajin.ui.ollama_helper import is_running


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
    statuses["openai"] = (
        _OK
        if settings.openai_api_key
        else ProviderStatus(available=False, reason="no API key")
    )
    statuses["ollama"] = (
        _OK
        if is_running(settings.ollama_base_url, timeout=0.5)
        else ProviderStatus(available=False, reason="Ollama offline")
    )

    return statuses
