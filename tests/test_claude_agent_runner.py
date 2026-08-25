"""Unit tests for the subscription-backed Claude Agent runner.

The SDK message → RunEvent translation and the tool-name bridging are pure and are
tested here with lightweight fakes, so no `claude` CLI or network is required.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from imajin.agent.providers.base import TextDelta, ToolUse, ToolUseStart
from imajin.agent.providers.claude_agent import (
    ClaudeAgentRunner,
    _explain_connect_failure,
    _flatten_tool_result,
    _map_usage,
    _strip_ns,
    _translate_message,
    subscription_available,
)
from imajin.agent.runner import ToolResult, TurnDone


def test_strip_ns_removes_mcp_prefix():
    assert _strip_ns("mcp__imajin__trace_neuron") == "trace_neuron"
    assert _strip_ns("already_bare") == "already_bare"


def test_flatten_tool_result_variants():
    assert _flatten_tool_result("plain") == "plain"
    assert _flatten_tool_result([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    # non-text blocks are JSON-encoded rather than dropped
    out = _flatten_tool_result([{"type": "image", "data": "x"}])
    assert "image" in out


def test_map_usage_keeps_known_int_keys_only():
    raw = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2, "junk": "no"}
    assert _map_usage(raw) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 2,
    }
    assert _map_usage(None) == {}


def test_translate_assistant_text_and_tool_use():
    id_to_name: dict[str, str] = {}
    msg = SimpleNamespace(
        content=[
            SimpleNamespace(text="thinking out loud"),
            SimpleNamespace(id="tu_1", name="mcp__imajin__do_thing", input={"a": 1}),
        ]
    )
    events = _translate_message(msg, id_to_name)

    assert isinstance(events[0], TextDelta) and events[0].text == "thinking out loud"
    assert isinstance(events[1], ToolUseStart) and events[1].name == "do_thing"
    assert isinstance(events[2], ToolUse)
    assert events[2].id == "tu_1" and events[2].input == {"a": 1}
    # the stripped name is remembered for the matching tool_result
    assert id_to_name == {"tu_1": "do_thing"}


def test_translate_tool_result_uses_remembered_name():
    id_to_name = {"tu_1": "do_thing"}
    msg = SimpleNamespace(
        content=[SimpleNamespace(tool_use_id="tu_1", content="42 objects", is_error=False)]
    )
    (event,) = _translate_message(msg, id_to_name)
    assert isinstance(event, ToolResult)
    assert event.name == "do_thing"
    assert event.output == "42 objects"
    assert event.is_error is False


def test_translate_result_message_to_turn_done():
    msg = SimpleNamespace(
        num_turns=1,
        session_id="sess_abc",
        subtype="success",
        stop_reason=None,
        usage={"input_tokens": 12, "output_tokens": 8},
    )
    (event,) = _translate_message(msg, {})
    assert isinstance(event, TurnDone)
    assert event.stop_reason == "end_turn"  # "success" is normalized
    assert event.total_usage == {"input_tokens": 12, "output_tokens": 8}


def test_translate_ignores_unknown_messages():
    # SystemMessage-like and StreamEvent-like objects yield nothing.
    assert _translate_message(SimpleNamespace(subtype="init", data={}), {}) == []
    assert _translate_message(SimpleNamespace(event="delta"), {}) == []


def test_subscription_available_returns_bool_and_reason():
    ok, reason = subscription_available()
    assert isinstance(ok, bool)
    assert ok or isinstance(reason, str)


def test_runner_lifecycle_flags():
    runner = ClaudeAgentRunner(model="sonnet", system_prompt="be helpful")
    assert runner.model == "sonnet"
    assert runner.name == "claude-agent"
    runner._session_id = "sess_1"
    runner.cancel()
    assert runner._cancelled is True
    runner.reset()
    assert runner._cancelled is False
    assert runner._session_id is None


def test_build_server_bridges_registry_tools():
    # Registering the tool package populates the registry the bridge reads from.
    import imajin.tools  # noqa: F401

    runner = ClaudeAgentRunner(model="sonnet", system_prompt="x")
    server, allowed = runner._build_server()
    assert server is not None
    assert allowed, "expected at least one Imajin tool to be bridged"
    assert all(name.startswith("mcp__imajin__") for name in allowed)
    # cached on second call
    assert runner._build_server()[1] is allowed


def test_turn_streams_events_and_reuses_one_persistent_connection(monkeypatch):
    """The runner connects once, reuses that client + loop across turns, and the
    close() teardown disconnects it — the fix for the per-turn-loop subprocess leak."""
    import imajin.tools  # noqa: F401  populate the registry the bridge reads

    from imajin.agent.providers import claude_agent as ca

    created: list = []

    class _FakeClient:
        def __init__(self, options=None):
            self.options = options
            self.connects = 0
            self.disconnects = 0
            self.queries: list[str] = []
            created.append(self)

        async def connect(self):
            self.connects += 1

        async def disconnect(self):
            self.disconnects += 1

        async def query(self, prompt):
            self.queries.append(prompt)

        async def receive_response(self):
            yield SimpleNamespace(content=[SimpleNamespace(text="hi there")])
            yield SimpleNamespace(
                num_turns=1,
                session_id="sess_1",
                subtype="success",
                stop_reason=None,
                usage={"input_tokens": 3, "output_tokens": 4},
            )

        async def interrupt(self):
            pass

    fake_sdk = SimpleNamespace(
        tool=lambda *a, **k: (lambda fn: fn),
        create_sdk_mcp_server=lambda **k: object(),
        ClaudeAgentOptions=lambda **k: SimpleNamespace(**k),
        ClaudeSDKClient=_FakeClient,
    )
    monkeypatch.setattr(ca, "_sdk", lambda: fake_sdk)

    runner = ca.ClaudeAgentRunner(model="sonnet", system_prompt="x")
    try:
        events1 = list(runner.turn("hello"))
        events2 = list(runner.turn("again"))
    finally:
        runner.close()

    assert any(isinstance(e, TextDelta) and e.text == "hi there" for e in events1)
    assert isinstance(events1[-1], TurnDone)
    assert isinstance(events2[-1], TurnDone)
    # One client, connected once, reused for both turns, disconnected by close().
    assert len(created) == 1
    assert created[0].connects == 1
    assert created[0].queries == ["hello", "again"]
    assert created[0].disconnects == 1
    assert runner._session_id == "sess_1"
    # The background loop/thread are gone after close().
    assert runner._loop is None and runner._thread is None


def test_turn_reports_connect_failure_in_stream(monkeypatch):
    from imajin.agent.providers import claude_agent as ca

    class _BoomClient:
        def __init__(self, options=None):
            pass

        async def connect(self):
            raise RuntimeError("no claude login")

        async def disconnect(self):
            pass

    fake_sdk = SimpleNamespace(
        tool=lambda *a, **k: (lambda fn: fn),
        create_sdk_mcp_server=lambda **k: object(),
        ClaudeAgentOptions=lambda **k: SimpleNamespace(**k),
        ClaudeSDKClient=_BoomClient,
    )
    monkeypatch.setattr(ca, "_sdk", lambda: fake_sdk)

    runner = ca.ClaudeAgentRunner(model="sonnet", system_prompt="x")
    try:
        events = list(runner.turn("hi"))
    finally:
        runner.close()

    assert any(isinstance(e, TextDelta) and "no claude login" in e.text for e in events)
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "error"


def test_system_prompt_is_passed_as_a_file_not_on_the_command_line(tmp_path):
    """The system prompt must not be an argv element.

    The SDK inlines `--system-prompt <text>`. Windows caps a whole command line
    at 32,767 characters and Imajin's prompt is ~30k before the ~3k of
    --allowedTools for its bridged tools, so inlining it overflowed:
    CreateProcess failed with WinError 206, Python raised FileNotFoundError, and
    the SDK reported it as "Claude Code not found at: <path>" for a claude.exe
    that was sitting right there.
    """
    prompt = "x" * 30_000
    runner = ClaudeAgentRunner(model="m", system_prompt=prompt)
    try:
        options = runner._build_options(server=object(), allowed=["mcp__imajin__a"])
        spec = options.system_prompt
        assert isinstance(spec, dict) and spec["type"] == "file"
        path = Path(spec["path"])
        assert path.read_text(encoding="utf-8") == prompt
        # Nothing prompt-sized is left inline.
        assert len(str(spec)) < 500
    finally:
        runner._cleanup_prompt_file()


def test_prompt_file_is_reused_then_cleaned_up():
    runner = ClaudeAgentRunner(model="m", system_prompt="hello")
    first = runner._write_prompt_file()
    assert first.exists()
    assert runner._write_prompt_file() == first  # reused, not re-created per turn
    runner._cleanup_prompt_file()
    assert not first.exists()
    assert runner._prompt_file is None
    runner._cleanup_prompt_file()  # idempotent


def test_connect_failure_explains_a_present_cli(tmp_path):
    """A 'not found' error naming a CLI that exists must not be taken at face value."""
    cli = tmp_path / "claude.exe"
    cli.write_bytes(b"MZ")

    explained = _explain_connect_failure(
        RuntimeError(f"Claude Code not found at: {cli}")
    )
    assert "exists but could not be launched" in str(explained)
    assert "32,767" in str(explained)


def test_connect_failure_is_left_alone_when_the_cli_really_is_missing(tmp_path):
    missing = tmp_path / "nope" / "claude.exe"
    original = RuntimeError(f"Claude Code not found at: {missing}")
    assert _explain_connect_failure(original) is original

    unrelated = RuntimeError("something else entirely")
    assert _explain_connect_failure(unrelated) is unrelated
