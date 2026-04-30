from __future__ import annotations

import os
from pathlib import Path

import pytest

from imajin.agent.providers.base import (
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)
from imajin.agent.runner import ToolResult, TurnDone

_KEY = os.environ.get("ANTHROPIC_API_KEY")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _KEY, reason="ANTHROPIC_API_KEY not set"),
]


def _build_runner(model: str = "claude-sonnet-4-6"):
    from imajin.agent.prompts import SYSTEM_PROMPT
    from imajin.agent.providers import AnthropicProvider
    from imajin.agent.runner import AgentRunner

    return AgentRunner(
        AnthropicProvider(api_key=_KEY, model=model),
        SYSTEM_PROMPT,
    )


def test_real_claude_calls_list_layers(viewer, tiny_ome_tiff: Path) -> None:
    from imajin.tools.files import load_file

    load_file(str(tiny_ome_tiff))

    runner = _build_runner()
    events = list(
        runner.turn("What image layers are currently loaded? Just enumerate them.")
    )

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    done = [e for e in events if isinstance(e, TurnDone)][-1]

    assert any(tr.name == "list_layers" for tr in tool_results), \
        f"expected list_layers call, got: {[tr.name for tr in tool_results]}"
    assert done.stop_reason in {"end_turn", "stop_sequence"}
    assert any(name in text for name in ("DAPI", "GFP", "TRITC")), \
        f"expected channel names mentioned, got: {text[:300]}"


def test_real_claude_two_turn_uses_cache(viewer, tiny_ome_tiff: Path) -> None:
    from imajin.tools.files import load_file

    load_file(str(tiny_ome_tiff))

    runner = _build_runner()
    list(runner.turn("What's loaded?"))
    events2 = list(runner.turn("Which channel has the highest mean intensity?"))

    done = [e for e in events2 if isinstance(e, TurnDone)][-1]
    cache_read = done.total_usage.get("cache_read_input_tokens", 0)
    assert cache_read > 0, (
        f"expected prompt-cache hit on second turn, usage={done.total_usage}"
    )
