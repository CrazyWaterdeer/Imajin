"""Compute availability of each LLM provider for the model picker.

imajin.agent.providers.registry.BACKEND_REGISTRY is the single source of truth
for which backend kinds exist and how to build them, but the five bare-name
probe calls below stay physically in this module rather than moving into that
registry: tests/test_provider_status.py and conftest.py's autouse
`_no_local_model_network` / `_no_codex_subscription` fixtures (which every
test in the suite inherits) patch probe_ollama / subscription_available /
codex_available as attributes of THIS module, relying on compute_statuses
resolving them as bare names against its own globals at call time. The
registry's own probes for claude-agent/codex-agent/ollama call back into
compute_statuses() below for exactly this reason -- see registry.py's module
docstring ("why probes delegate") for the full explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imajin.agent.local_models import probe_ollama
from imajin.agent.providers.claude_agent import subscription_available
from imajin.agent.providers.codex_agent import codex_available


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
    # Subscription-backed like claude-agent above -- no API key, just a
    # logged-in `codex` CLI. Registering this is not optional polish: a kind
    # with no entry here used to render as selectable in the picker (see
    # chat_dock._UNREGISTERED_STATUS), so skipping this line would ship a
    # "Codex (subscription)" row that looks fine and fails on first send.
    codex_ok, codex_reason = codex_available()
    statuses["codex-agent"] = (
        _OK if codex_ok else ProviderStatus(available=False, reason=codex_reason)
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
