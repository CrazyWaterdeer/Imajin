from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
)
from imajin.agent.specialists.report_writer import (
    consult_report_writer_via_provider,
)


@dataclass
class _ScriptedProvider:
    name: str = "scripted"
    model: str = "fake-1"
    last_tools: list[dict[str, Any]] = field(default_factory=list)
    last_user_text: str = ""

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        self.last_tools = tools
        self.last_user_text = messages[-1]["content"][0]["text"]
        yield TextDelta(text="Confocal images were segmented with Cellpose-SAM. ")
        yield TextDelta(text="Per-cell intensities were extracted with regionprops_table.")
        yield Stop(reason="end_turn")


def _records() -> list[dict[str, Any]]:
    return [
        {
            "tool": "cellpose_sam",
            "inputs": {"image_layer": "img", "channel": 0, "do_3D": False, "model": "cpsam"},
            "duration_s": 5.0,
            "ok": True,
        },
        {
            "tool": "measure_intensity",
            "inputs": {"labels_layer": "img_masks", "image_layer": "img", "channels": [1]},
            "duration_s": 0.5,
            "ok": True,
        },
        {
            "tool": "rolling_ball_background",
            "inputs": {"layer": "img", "radius": 25},
            "duration_s": 2.0,
            "ok": False,  # failed → must be excluded
        },
    ]


def test_report_writer_receives_no_tools() -> None:
    provider = _ScriptedProvider()
    text = consult_report_writer_via_provider(provider, _records(), style="paper")
    assert text.startswith("Confocal")
    assert provider.last_tools == []
    # Failed entries are stripped before being shown
    assert "rolling_ball" not in provider.last_user_text


def test_report_writer_passes_style_in_prompt() -> None:
    provider = _ScriptedProvider()
    consult_report_writer_via_provider(provider, _records(), style="protocol")
    assert "protocol" in provider.last_user_text


def test_report_writer_rejects_bad_style() -> None:
    provider = _ScriptedProvider()
    with pytest.raises(ValueError, match="style must be"):
        consult_report_writer_via_provider(provider, _records(), style="essay")
