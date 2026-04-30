from __future__ import annotations

from typing import Any

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)
from imajin.agent.runner import AgentRunner, ToolResult, TurnDone
from imajin.tools import tool
from imajin.tools import registry as registry_mod


class _ScriptedProvider:
    name = "scripted"
    model = "scripted-test"

    def __init__(self, scripts: list[list[Event]]) -> None:
        self._scripts = scripts
        self.calls: list[tuple[list, list, str]] = []

    def stream(self, messages, tools, system):
        idx = len(self.calls)
        self.calls.append((list(messages), list(tools), system))
        events = self._scripts[idx]
        for e in events:
            yield e


import pytest


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(registry_mod._REGISTRY)
    registry_mod._REGISTRY.clear()
    yield
    registry_mod._REGISTRY.clear()
    registry_mod._REGISTRY.update(saved)


def test_runner_dispatches_tool_and_continues() -> None:
    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    provider = _ScriptedProvider(
        [
            [
                TextDelta(text="Let me compute that."),
                ToolUseStart(id="tu_1", name="add"),
                ToolUse(id="tu_1", name="add", input={"a": 2, "b": 3}),
                Stop(reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5}),
            ],
            [
                TextDelta(text="The answer is 5."),
                Stop(reason="end_turn", usage={"input_tokens": 12, "output_tokens": 4}),
            ],
        ]
    )

    runner = AgentRunner(provider, "test prompt")
    events = list(runner.turn("compute 2+3"))

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    done = [e for e in events if isinstance(e, TurnDone)]

    assert "Let me compute" in text
    assert "answer is 5" in text
    assert len(tool_results) == 1
    assert tool_results[0].output == 5
    assert tool_results[0].is_error is False
    assert done[-1].stop_reason == "end_turn"
    assert done[-1].total_usage["input_tokens"] == 22
    assert provider.calls[1][0][-1]["role"] == "user"
    assert provider.calls[1][0][-1]["content"][0]["type"] == "tool_result"


def test_runner_records_error_on_tool_exception() -> None:
    @tool()
    def boom(x: int) -> int:
        raise ValueError("nope")

    provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu_1", name="boom"),
                ToolUse(id="tu_1", name="boom", input={"x": 1}),
                Stop(reason="tool_use"),
            ],
            [
                TextDelta(text="Sorry, that failed."),
                Stop(reason="end_turn"),
            ],
        ]
    )
    runner = AgentRunner(provider, "test")
    events = list(runner.turn("call boom"))

    errors = [e for e in events if isinstance(e, ToolResult) and e.is_error]
    assert len(errors) == 1
    assert "nope" in str(errors[0].output)


def test_runner_max_loops_terminates() -> None:
    @tool()
    def noop() -> int:
        return 0

    looping = [
        ToolUseStart(id="tu", name="noop"),
        ToolUse(id="tu", name="noop", input={}),
        Stop(reason="tool_use"),
    ]
    provider = _ScriptedProvider([looping] * 20)

    runner = AgentRunner(provider, "test", max_loops=3)
    events = list(runner.turn("loop forever"))
    done = [e for e in events if isinstance(e, TurnDone)]
    assert done[-1].stop_reason == "max_loops"


def test_runner_cancellation_stops_dispatch() -> None:
    @tool()
    def slow_thing() -> int:
        return 1

    provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu", name="slow_thing"),
                ToolUse(id="tu", name="slow_thing", input={}),
                Stop(reason="tool_use"),
            ],
            [TextDelta(text="ok"), Stop(reason="end_turn")],
        ]
    )
    runner = AgentRunner(provider, "test")
    runner.cancel()
    events = list(runner.turn("anything"))
    done = [e for e in events if isinstance(e, TurnDone)]
    assert done[-1].stop_reason == "cancelled"
