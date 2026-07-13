"""Gate for the 'Helping vs acting' system-prompt block.

The deterministic tests pin the prompt *contract* (block present, placed after
bias-to-action and before batch progress, carrying the key precedence language) so a
future edit that guts the semantics fails CI without needing an LLM. The behavioural
evals (opt-in, `integration` marker) exercise the real model on the trace: an
analysis request must not call `get_help`, and an orientation question must."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imajin.agent.prompts import SYSTEM_PROMPT

# --- deterministic contract (CI gate, no LLM) ---------------------------------------


def test_help_block_present_and_placed():
    i_bias = SYSTEM_PROMPT.find("Bias to action")
    i_help = SYSTEM_PROMPT.find("Helping vs acting")
    i_batch = SYSTEM_PROMPT.find("Batch progress")
    assert i_bias != -1 and i_help != -1 and i_batch != -1
    # subordinate to bias-to-action, above batch progress
    assert i_bias < i_help < i_batch


def test_help_block_precedence_language():
    block = SYSTEM_PROMPT[SYSTEM_PROMPT.find("Helping vs acting"):SYSTEM_PROMPT.find("Batch progress")]
    assert "get_help" in block
    assert "onboarding" in block.lower()
    # the load-bearing rule: a question-shaped request to act must still act
    assert "phrased as a question" in block
    # explicitly subordinate to bias-to-action
    assert "bias to action" in block.lower()


# --- behavioural evals (opt-in; trace-based) ----------------------------------------

_KEY = os.environ.get("ANTHROPIC_API_KEY")
# Pin the model so the eval is reproducible; document a modest retry allowance for LLM
# nondeterminism if this proves flaky in a scheduled run.
_EVAL_MODEL = "claude-sonnet-4-6"


def _build_runner():
    from imajin.agent.providers import AnthropicProvider
    from imajin.agent.runner import AgentRunner

    return AgentRunner(AnthropicProvider(api_key=_KEY, model=_EVAL_MODEL), SYSTEM_PROMPT)


def _tool_names(prompt: str) -> list[str]:
    from imajin.agent.runner import ToolResult

    events = list(_build_runner().turn(prompt))
    return [e.name for e in events if isinstance(e, ToolResult)]


def _is_analysis(name: str) -> bool:
    return name == "list_layers" or "segment" in name or "measure" in name or "count" in name


integration = pytest.mark.integration
skip_no_key = pytest.mark.skipif(not _KEY, reason="ANTHROPIC_API_KEY not set")


@integration
@skip_no_key
@pytest.mark.parametrize(
    "prompt",
    ["measure Ch2 intensity", "how do I measure Ch2 intensity?", "세포 찾아"],
)
def test_analysis_request_does_not_call_get_help(prompt, viewer, tiny_ome_tiff: Path):
    from imajin.tools.files import load_file

    load_file(str(tiny_ome_tiff))
    names = _tool_names(prompt)
    assert "get_help" not in names, f"{prompt!r} wrongly routed to help: {names}"
    assert any(_is_analysis(n) for n in names), f"{prompt!r} ran no analysis: {names}"


@integration
@skip_no_key
@pytest.mark.parametrize(
    "prompt",
    ["what can you do?", "how do I get started?", "don't run anything, just show me how to start"],
)
def test_orientation_question_calls_get_help(prompt):
    # no data loaded — pure orientation
    names = _tool_names(prompt)
    assert "get_help" in names, f"{prompt!r} did not call get_help: {names}"
    assert not any(_is_analysis(n) for n in names), f"{prompt!r} ran an analysis: {names}"
