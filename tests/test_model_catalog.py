"""Tests for latest-model resolution (Claude API + OpenAI tiers).

Pure selection helpers and the cached resolver are exercised with fake clients, so
no network or API key is required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from imajin.agent import model_catalog as mc


@pytest.fixture(autouse=True)
def _clear_cache():
    mc.clear_cache()
    yield
    mc.clear_cache()


class _FakeModels:
    def __init__(self, items, counter):
        self._items = items
        self._counter = counter

    def list(self):
        self._counter[0] += 1
        return list(self._items)


class _FakeClient:
    def __init__(self, items, counter, raise_exc=None):
        self.models = _FakeModels(items, counter)
        self._raise = raise_exc

    def with_options(self, **_kwargs):
        if self._raise is not None:
            raise self._raise
        return self


def test_pick_newest_anthropic_by_created_at():
    models = [
        SimpleNamespace(id="claude-3-5-sonnet-20241022", created_at="2024-10-22"),
        SimpleNamespace(id="claude-sonnet-4-5", created_at="2025-09-29"),
        SimpleNamespace(id="claude-sonnet-4-6", created_at="2026-05-01"),
        SimpleNamespace(id="claude-opus-4-7", created_at="2026-06-01"),
    ]
    assert mc._pick_newest_anthropic(models, "sonnet", "fallback") == "claude-sonnet-4-6"
    assert mc._pick_newest_anthropic(models, "opus", "fallback") == "claude-opus-4-7"
    # no match → fallback
    assert mc._pick_newest_anthropic(models, "haiku", "claude-haiku-4-5") == "claude-haiku-4-5"


def test_pick_newest_openai_excludes_non_flagship():
    models = [
        SimpleNamespace(id="gpt-5", created=100),
        SimpleNamespace(id="gpt-5.1", created=300),
        SimpleNamespace(id="gpt-5-mini", created=400),  # excluded despite newest
        SimpleNamespace(id="text-embedding-3-large", created=500),  # excluded
        SimpleNamespace(id="gpt-4o-audio", created=250),  # excluded
    ]
    assert mc._pick_newest_openai(models, "gpt-5") == "gpt-5.1"
    # nothing flagship → fallback
    only_specialty = [SimpleNamespace(id="gpt-5-mini", created=1)]
    assert mc._pick_newest_openai(only_specialty, "gpt-5") == "gpt-5"


def test_resolve_anthropic_caches_and_upgrades():
    counter = [0]
    client = _FakeClient(
        [
            SimpleNamespace(id="claude-sonnet-4-5", created_at="2025-09-29"),
            SimpleNamespace(id="claude-sonnet-4-6", created_at="2026-05-01"),
        ],
        counter,
    )
    assert mc.resolve_anthropic_model(client, "sonnet") == "claude-sonnet-4-6"
    # second call served from cache — no extra network hit
    assert mc.resolve_anthropic_model(client, "sonnet") == "claude-sonnet-4-6"
    assert counter[0] == 1


def test_resolve_anthropic_falls_back_on_error():
    counter = [0]
    client = _FakeClient([], counter, raise_exc=RuntimeError("network down"))
    # falls back to the known-good default rather than raising
    assert mc.resolve_anthropic_model(client, "sonnet") == "claude-sonnet-4-6"


def test_resolve_openai_falls_back_on_error():
    counter = [0]
    client = _FakeClient([], counter, raise_exc=RuntimeError("network down"))
    assert mc.resolve_openai_model(client, "gpt") == "gpt-5"
