"""Deterministic (no-LLM) tests for the onboarding `get_help` tool and its coupling to
``docs/getting_started.md``. This is the CI gate for the feature."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import imajin.tools  # noqa: F401  -- triggers @tool registration
from imajin.tools.help import (
    GUIDE_URL,
    _TOPICS,
    _github_slug,
    _match,
    get_help,
)
from imajin.tools.registry import get_tool, tools_for_anthropic

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "getting_started.md"
README = REPO_ROOT / "README.md"

# Broad, cross-cutting words that must never be a standalone keyword (they would hijack
# an ordered first-match). Multi-word keywords that merely contain one are fine.
DENYLIST = ("analysis", "analyse", "cells", "compare", "data", "image", "channel")

HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")


def _guide_headings() -> list[str]:
    """The `##` headings of the guide, ignoring headings inside fenced code blocks."""
    headings: list[str] = []
    in_fence = False
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            headings.append(m.group(1))
    return headings


# --- shape --------------------------------------------------------------------------


def test_overview_shape():
    out = get_help()
    assert out["guide_url"] == GUIDE_URL
    assert out["topics"], "overview must list topics"
    for t in out["topics"]:
        assert t["id"] and t["title"] and t["summary"]


@pytest.mark.parametrize("empty", [None, "", "   ", "\t"])
def test_empty_query_is_overview(empty):
    out = get_help(empty)
    # overview shape: no `matched` key, carries topics + guide_url, never raises
    assert "matched" not in out
    assert out["guide_url"] == GUIDE_URL
    assert out["topics"]
    assert _match(empty) is None


def test_no_match_shape():
    out = get_help("nonexistent-xyz-topic")
    assert out["matched"] is False
    assert out["guide_url"] == GUIDE_URL
    assert out["topics"]


# --- registration -------------------------------------------------------------------


def test_registered_and_llm_visible():
    entry = get_tool("get_help")
    assert entry.llm is True
    assert entry.manual is False
    names = {t["name"] for t in tools_for_anthropic()}
    assert "get_help" in names


def test_result_is_json_serialisable():
    for arg in (None, "install", "nope"):
        json.dumps(get_help(arg), default=str)  # must not raise


# --- routing ------------------------------------------------------------------------

ROUTING = {
    # install_and_run
    "how do I install": "install_and_run",
    "install": "install_and_run",
    "uv sync": "install_and_run",
    "get started": "install_and_run",
    # open_your_data
    "open my data": "open_your_data",
    "how do I load a file": "open_your_data",
    "open file": "open_your_data",
    # tell_imajin_your_channels
    "how do I set channel roles": "tell_imajin_your_channels",
    "counterstain": "tell_imajin_your_channels",
    # first_analysis
    "measure": "first_analysis",
    "how do I measure intensity": "first_analysis",
    "find cells": "first_analysis",
    "segment the objects": "first_analysis",
    # reading_results
    "where are the results": "reading_results",
    "save results": "reading_results",
    # two_ways_to_drive_it
    "chat dock or manual dock": "two_ways_to_drive_it",
    "gui": "two_ways_to_drive_it",
    # if_something_goes_wrong
    "I get an error": "if_something_goes_wrong",
    "gpu not working": "if_something_goes_wrong",
    "how do I run offline": "if_something_goes_wrong",
    # asking_for_help
    "help": "asking_for_help",
    # go_deeper
    "advanced docs": "go_deeper",
    "learn more": "go_deeper",
    # what_is_imajin
    "what is imajin": "what_is_imajin",
    "give me an overview": "what_is_imajin",
}


@pytest.mark.parametrize("phrase,expected_id", list(ROUTING.items()))
def test_routing_table(phrase, expected_id):
    out = get_help(phrase)
    assert out.get("matched") is True, f"{phrase!r} did not match"
    assert out["id"] == expected_id
    assert out["url"] == f"{GUIDE_URL}#{next(t.anchor for t in _TOPICS if t.id == expected_id)}"


def test_every_keyword_resolves_to_its_owner():
    for t in _TOPICS:
        for kw in t.keywords:
            out = get_help(kw)
            assert out.get("matched") is True, f"keyword {kw!r} of {t.id} did not match"
            assert out["id"] == t.id, f"keyword {kw!r} routed to {out['id']}, expected {t.id}"


def test_no_cross_topic_substring_collision():
    def tokens(t):
        return (t.id, t.title.casefold(), *(k.casefold() for k in t.keywords))

    for x in _TOPICS:
        for y in _TOPICS:
            if x.id == y.id:
                continue  # same-topic overlaps (e.g. measure < measure intensity) are fine
            for a in tokens(x):
                for b in tokens(y):
                    assert a not in b, f"token {a!r} ({x.id}) is a substring of {b!r} ({y.id})"


# --- keyword hygiene ----------------------------------------------------------------


def test_no_keyword_equals_broad_word():
    for t in _TOPICS:
        for kw in t.keywords:
            assert kw.casefold() not in DENYLIST, f"{kw!r} in {t.id} is a banned broad keyword"


@pytest.mark.parametrize("word", DENYLIST)
def test_broad_word_routes_nowhere(word):
    # a bare broad word must not resolve to any section (via keyword, id, or title)
    assert _match(word) is None
    assert get_help(word)["matched"] is False


# --- doc <-> topic coupling ---------------------------------------------------------


def test_section_topic_bijection():
    headings = _guide_headings()
    assert len(headings) == len(_TOPICS), "guide section count != _TOPICS count"
    ids = [t.id for t in _TOPICS]
    anchors = [t.anchor for t in _TOPICS]
    assert len(set(ids)) == len(ids), "duplicate topic id"
    assert len(set(anchors)) == len(anchors), "duplicate topic anchor"
    assert {_github_slug(h) for h in headings} == set(anchors)


def test_heading_contract_and_anchor_equality():
    headings = _guide_headings()
    slug_charset = re.compile(r"^[A-Za-z0-9 ]+$")
    for h in headings:
        assert slug_charset.match(h), f"heading {h!r} breaks the ASCII/punctuation-light contract"
    # every topic anchor equals the slug of its own title
    for t in _TOPICS:
        assert t.anchor == _github_slug(t.title)


# --- link presence ------------------------------------------------------------------


def test_readme_links_to_new_docs():
    text = README.read_text(encoding="utf-8")
    assert "docs/getting_started.md" in text
    assert "docs/features.md" in text


def test_guide_relative_md_links_resolve():
    text = GUIDE.read_text(encoding="utf-8")
    targets = MD_LINK_RE.findall(text)
    assert targets, "expected relative .md links in the guide"
    for rel in targets:
        assert (GUIDE.parent / rel).resolve().is_file(), f"broken guide link: {rel}"
