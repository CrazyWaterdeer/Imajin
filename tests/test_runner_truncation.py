from __future__ import annotations

from typing import Any

import pytest

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)
from imajin.agent.runner import AgentRunner, TurnDone

# Covers two runner.py bugs:
#
#   1. THE ORPHAN TOOL_USE BUG: when the provider stops for a reason other
#      than "tool_use" (notably "length") while assistant_blocks still holds
#      an unfinished tool_use, the old code committed it anyway. Every later
#      turn then sends a message list with a tool_use that has no matching
#      tool_result, which Anthropic rejects outright and local servers
#      mishandle -- permanently corrupting the conversation, not just that
#      turn.
#   2. Losing already-streamed text on a mid-stream exception: the old code
#      re-raised straight out of turn() without ever appending the partial
#      assistant text to self.messages, so a transient failure after minutes
#      of local-model generation threw the response away instead of keeping
#      it in history.


class _ScriptedProvider:
    """Replays one canned event list per call to stream() (one call per
    AgentRunner provider-loop iteration)."""

    name = "scripted"
    model = "scripted-test"

    def __init__(self, scripts: list[list[Event]]) -> None:
        self._scripts = scripts
        self.calls: list[tuple[list, list, str]] = []

    def stream(self, messages, tools, system):
        idx = len(self.calls)
        self.calls.append((list(messages), list(tools), system))
        yield from self._scripts[idx]


class _RecordingProvider:
    """Records the message list it is called with and replays one script,
    regardless of how many times it is called -- used to inspect exactly
    what a *second* turn sends after the first turn mutated runner.messages.
    """

    name = "recording"
    model = "recording-test"

    def __init__(self, script: list[Event]) -> None:
        self._script = script
        self.received: list[list[dict[str, Any]]] = []

    def stream(self, messages, tools, system):
        self.received.append(list(messages))
        yield from self._script


def _tool_use_ids(content: Any) -> set[str]:
    if not isinstance(content, list):
        return set()
    return {b["id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_use"}


def _tool_result_ids(content: Any) -> set[str]:
    if not isinstance(content, list):
        return set()
    return {
        b["tool_use_id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
    }


def _assert_no_orphan_tool_use(messages: list[dict[str, Any]]) -> None:
    """Every tool_use in `messages` must be immediately followed (the very
    next message in the list) by a tool_result covering it -- the invariant
    every provider API enforces, and the one the orphan-tool_use bug broke.
    """
    for i, msg in enumerate(messages):
        want = _tool_use_ids(msg.get("content"))
        if not want:
            continue
        assert i + 1 < len(messages), f"message {i} has tool_use with nothing after it"
        have = _tool_result_ids(messages[i + 1].get("content"))
        assert want <= have, f"message {i} has orphan tool_use ids {want - have}"


# --- bug 1: a non-"tool_use" stop must not commit an orphan tool_use -------


def test_length_cutoff_mid_tool_call_drops_orphan_tool_use() -> None:
    provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu_1", name="whatever"),
                ToolUse(id="tu_1", name="whatever", input={"x": 1}),
                Stop(reason="length"),
            ],
        ]
    )
    runner = AgentRunner(provider, "test")
    events = list(runner.turn("do the thing"))

    _assert_no_orphan_tool_use(runner.messages)

    done = [e for e in events if isinstance(e, TurnDone)]
    assert done[-1].stop_reason == "length"

    explanation = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert explanation, "the cut-off must be explained to the user, not silent"
    assert "retry" in explanation.lower()


def test_length_cutoff_keeps_text_produced_before_the_cut() -> None:
    provider = _ScriptedProvider(
        [
            [
                TextDelta(text="Let me look at the files first."),
                ToolUseStart(id="tu_1", name="whatever"),
                ToolUse(id="tu_1", name="whatever", input={}),
                Stop(reason="length"),
            ],
        ]
    )
    runner = AgentRunner(provider, "test")
    list(runner.turn("go"))

    assistant_msgs = [m for m in runner.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == [
        {"type": "text", "text": "Let me look at the files first."}
    ]


def test_length_cutoff_with_only_a_tool_use_commits_nothing() -> None:
    provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu_1", name="whatever"),
                ToolUse(id="tu_1", name="whatever", input={}),
                Stop(reason="length"),
            ],
        ]
    )
    runner = AgentRunner(provider, "test")
    list(runner.turn("go"))

    # Nothing survives to commit (spec: "if nothing is left, commit
    # nothing") -- only the original user message is in history.
    assert len(runner.messages) == 1
    assert runner.messages[0]["role"] == "user"


def test_second_turn_after_cutoff_sends_a_well_formed_message_list() -> None:
    cutoff_provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu_1", name="whatever"),
                ToolUse(id="tu_1", name="whatever", input={"x": 1}),
                Stop(reason="length"),
            ],
        ]
    )
    runner = AgentRunner(cutoff_provider, "test")
    list(runner.turn("do the thing"))

    recorder = _RecordingProvider([TextDelta(text="ok"), Stop(reason="end_turn")])
    runner.provider = recorder
    list(runner.turn("try again"))

    sent = recorder.received[0]
    _assert_no_orphan_tool_use(sent)
    # And the retry actually reached the provider as plain user text.
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"][0]["text"] == "try again"


def test_stop_reason_other_than_length_also_drops_orphan_tool_use() -> None:
    # Not just "length" -- any non-"tool_use" stop with a dangling tool_use
    # must be handled the same way.
    provider = _ScriptedProvider(
        [
            [
                ToolUseStart(id="tu_1", name="whatever"),
                ToolUse(id="tu_1", name="whatever", input={}),
                Stop(reason="content_filter"),
            ],
        ]
    )
    runner = AgentRunner(provider, "test")
    events = list(runner.turn("go"))
    _assert_no_orphan_tool_use(runner.messages)
    done = [e for e in events if isinstance(e, TurnDone)]
    assert done[-1].stop_reason == "content_filter"


# --- bug 2: a mid-stream exception must not lose text already shown -------


def test_mid_stream_exception_commits_partial_text_before_raising() -> None:
    class _DyingProvider:
        name = "dying"
        model = "dying-test"

        def stream(self, messages, tools, system):
            yield TextDelta(text="Partial answer before ")
            yield TextDelta(text="the connection dropped.")
            raise ConnectionError("connection reset by peer")

    runner = AgentRunner(_DyingProvider(), "test")
    seen_text: list[str] = []
    with pytest.raises(ConnectionError):
        for event in runner.turn("hello"):
            if isinstance(event, TextDelta):
                seen_text.append(event.text)

    assert "".join(seen_text) == "Partial answer before the connection dropped."
    assert runner.messages[-1]["role"] == "assistant"
    assert runner.messages[-1]["content"] == [
        {"type": "text", "text": "Partial answer before the connection dropped."}
    ]


def test_mid_stream_exception_drops_incomplete_tool_use_too() -> None:
    class _DyingMidToolProvider:
        name = "dying"
        model = "dying-test"

        def stream(self, messages, tools, system):
            yield ToolUseStart(id="tu_1", name="whatever")
            yield ToolUse(id="tu_1", name="whatever", input={})
            raise RuntimeError("socket closed")

    runner = AgentRunner(_DyingMidToolProvider(), "test")
    with pytest.raises(RuntimeError, match="socket closed"):
        list(runner.turn("hello"))

    _assert_no_orphan_tool_use(runner.messages)
    # Nothing but the tool_use was produced, so nothing is left to commit.
    assert len(runner.messages) == 1


def test_mid_stream_exception_with_no_output_at_all_commits_nothing() -> None:
    class _ImmediatelyDyingProvider:
        name = "dying"
        model = "dying-test"

        def stream(self, messages, tools, system):
            raise TimeoutError("no response from server")
            yield  # pragma: no cover - generator marker, never reached

    runner = AgentRunner(_ImmediatelyDyingProvider(), "test")
    with pytest.raises(TimeoutError):
        list(runner.turn("hello"))

    assert len(runner.messages) == 1
    assert runner.messages[0]["role"] == "user"


def test_mid_stream_exception_is_not_confused_with_context_limit() -> None:
    # A generic connection failure must be committed-and-reraised (bug 2's
    # fix), not swallowed by the unrelated context-limit-recovery path.
    class _DyingProvider:
        name = "dying"
        model = "dying-test"

        def stream(self, messages, tools, system):
            yield TextDelta(text="partial")
            raise OSError("connection reset")

    runner = AgentRunner(_DyingProvider(), "test")
    with pytest.raises(OSError, match="connection reset"):
        list(runner.turn("hello"))
    assert runner.messages[-1]["content"] == [{"type": "text", "text": "partial"}]
