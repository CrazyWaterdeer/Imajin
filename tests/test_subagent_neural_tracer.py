from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)
from imajin.agent.specialists.neural_tracer import (
    consult_neural_tracer_via_provider,
)
from imajin.tools import trace


@dataclass
class _ScriptedProvider:
    """Replays a list of pre-baked event sequences (one per turn)."""

    name: str = "scripted"
    model: str = "fake-1"
    turns: list[list[Event]] = field(default_factory=list)
    _turn_idx: int = 0

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        if self._turn_idx >= len(self.turns):
            yield Stop(reason="end_turn")
            return
        events = self.turns[self._turn_idx]
        self._turn_idx += 1
        yield from events


def _y_mask() -> np.ndarray:
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def test_subagent_runs_skeletonize_then_branch_metrics(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_y_mask(), name="ymask")

    provider = _ScriptedProvider(
        turns=[
            [
                ToolUseStart(id="tu_1", name="skeletonize"),
                ToolUse(id="tu_1", name="skeletonize", input={"layer": "ymask"}),
                Stop(reason="tool_use"),
            ],
            [
                TextDelta(text="OK so the skeleton has "),
                TextDelta(text="multiple branches. "),
                ToolUseStart(id="tu_2", name="extract_branch_metrics"),
                ToolUse(
                    id="tu_2",
                    name="extract_branch_metrics",
                    input={"skeleton_id": "skel_0_ymask"},
                ),
                Stop(reason="tool_use"),
            ],
            [
                TextDelta(text="Done — "),
                TextDelta(text="3 branches detected."),
                Stop(reason="end_turn", usage={"input_tokens": 10, "output_tokens": 5}),
            ],
        ]
    )

    result = consult_neural_tracer_via_provider(
        provider, "이 mask의 분기 구조 분석해줘", target_layer="ymask"
    )

    assert result.stop_reason == "end_turn"
    assert "Done" in result.text
    names = [c["name"] for c in result.tool_calls]
    assert names == ["skeletonize", "extract_branch_metrics"]
    assert all(c["ok"] for c in result.tool_calls)
    assert result.tool_calls[0]["output"]["skeleton_id"] == "skel_0_ymask"
    assert result.tool_calls[1]["output"]["n_branches"] >= 3


def test_subagent_only_sees_specialist_tools() -> None:
    """Sanity: the tools list passed to the specialist provider is filtered."""
    from imajin.tools import tools_for_anthropic

    public_names = {t["name"] for t in tools_for_anthropic()}
    specialist_names = {t["name"] for t in tools_for_anthropic(subagent="neural_tracer")}

    assert public_names.isdisjoint(specialist_names)
    assert "skeletonize" in specialist_names
    assert "skeletonize" not in public_names
    assert "consult_neural_tracer" in public_names
    assert "consult_neural_tracer" not in specialist_names
