"""Unit tests for the subscription-backed Codex Agent runner.

The JSONL -> RunEvent translation is tested against the REAL stdout captured
from 5 live `codex exec --json` runs during this slice's probe (see
codex_agent.py's module docstring for the ToS/architecture context, and the
probe evidence this file was built from). Fixtures are embedded verbatim
below rather than read from the probe's scratchpad path, which is
session-local and would not exist for another run of this suite.

Everything here uses a fake `subprocess.Popen` — no `codex` CLI, network, or
subscription quota is ever touched. `imajin.agent.mcp_bridge` (a sibling
slice) is never imported either: tests either pre-seed `runner._bridge` with
a trivial stand-in (for tests about turn()'s own translation/process
plumbing) or monkeypatch the module's lazy `_bridge_cls()` loader (for the
one test that checks _ensure_bridge()'s wiring against the pinned
ImajinMcpBridge constructor shape).
"""
from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from imajin.agent.providers.base import TextDelta, ToolUse, ToolUseStart
from imajin.agent.providers.codex_agent import (
    CodexAgentRunner,
    _codex_version,
    _failure_message,
    _flatten_mcp_result,
    _looks_unauthenticated,
    _parse_version,
    _terminate_process,
    _tool_call_outcome,
    _translate_event,
    codex_available,
)
from imajin.agent.runner import ToolResult, TurnDone

_FAKE_CODEX_PATH = "/fake/bin/codex"

# ---------------------------------------------------------------------------
# Fixtures: raw stdout JSONL captured from 5 live `codex exec --json` runs.
# The curly quotes in some agent_message text are codex's own generated
# punctuation, copied verbatim from the capture — not typos introduced here.
# ---------------------------------------------------------------------------

APPROVE_LINES = [
    '{"type":"thread.started","thread_id":"01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"I’ll call the requested tool directly."}}',
    '{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"hello"},"result":null,"error":null,'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"hello"},"result":{"content":[{"type":"text",'
    '"text":"HELLO"}],"structured_content":null},"error":null,"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"HELLO"}}',
    '{"type":"turn.completed","usage":{"input_tokens":47759,"cached_input_tokens":36992,'
    '"output_tokens":136,"reasoning_output_tokens":27}}',
]
APPROVE_JSONL = "\n".join(APPROVE_LINES) + "\n"

NOAPPROVAL_LINES = [
    '{"type":"thread.started","thread_id":"01a08ee2-c5ea-7150-90da-50d752743dd6"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"I’m calling the requested ping tool now."}}',
    '{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"hello"},"result":null,"error":null,'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"hello"},"result":null,'
    '"error":{"message":"user cancelled MCP tool call"},"status":"failed"}}',
    '{"type":"item.completed","item":{"id":"item_2","type":"agent_message",'
    '"text":"user cancelled MCP tool call"}}',
    '{"type":"turn.completed","usage":{"input_tokens":32485,"cached_input_tokens":21248,'
    '"output_tokens":154,"reasoning_output_tokens":53}}',
]
NOAPPROVAL_JSONL = "\n".join(NOAPPROVAL_LINES) + "\n"

RESUME_LINES = [
    '{"type":"thread.started","thread_id":"01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"}',
    '{"type":"turn.started"}',
    '{"type":"item.started","item":{"id":"item_0","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"world"},"result":null,"error":null,'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_0","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"world"},"result":{"content":[{"type":"text",'
    '"text":"WORLD"}],"structured_content":null},"error":null,"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"WORLD"}}',
    '{"type":"turn.completed","usage":{"input_tokens":80002,"cached_input_tokens":68736,'
    '"output_tokens":232,"reasoning_output_tokens":27}}',
]
RESUME_JSONL = "\n".join(RESUME_LINES) + "\n"

ERRORPATH_LINES = [
    '{"type":"thread.started","thread_id":"01a08ee4-2350-7c61-a605-58e6615578ea"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"I’ll make the two ping calls in the requested order."}}',
    '{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"boom"},"result":null,"error":null,'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"boom"},"result":{"content":[{"type":"text",'
    '"text":"imajin tool blew up: no image loaded"}],"structured_content":null},"error":null,'
    '"status":"failed"}}',
    '{"type":"item.started","item":{"id":"item_2","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"ok"},"result":null,"error":null,'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_2","type":"mcp_tool_call","server":"probe",'
    '"tool":"imajin_ping","arguments":{"word":"ok"},"result":{"content":[{"type":"text",'
    '"text":"OK"}],"structured_content":null},"error":null,"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_3","type":"agent_message",'
    '"text":"“boom” failed: no image loaded; “ok” succeeded: OK."}}',
    '{"type":"turn.completed","usage":{"input_tokens":47920,"cached_input_tokens":36864,'
    '"output_tokens":217,"reasoning_output_tokens":46}}',
]
ERRORPATH_JSONL = "\n".join(ERRORPATH_LINES) + "\n"

UNAUTHENTICATED_LINES = [
    '{"type":"thread.started","thread_id":"01a08edf-4e2a-7ef2-aa60-e0e9779dc0e9"}',
    '{"type":"turn.started"}',
    '{"type":"error","message":"Reconnecting... 2/5 (unexpected status 401 Unauthorized: '
    'Missing bearer or basic authentication in header, url: wss://api.openai.com/v1/responses,'
    ' cf-ray: a39429bbcbdeeaaf-ICN)"}',
    '{"type":"error","message":"Reconnecting... 5/5 (unexpected status 401 Unauthorized: '
    'Missing bearer or basic authentication in header, url: wss://api.openai.com/v1/responses,'
    ' cf-ray: a39429d0cc69ea2d-ICN)"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Falling back '
    'from WebSockets to HTTPS transport. unexpected status 401 Unauthorized: Missing bearer or'
    ' basic authentication in header, url: wss://api.openai.com/v1/responses, cf-ray:'
    ' a39429e36810ea2b-ICN"}}',
    '{"type":"error","message":"Reconnecting... 5/5 (unexpected status 401 Unauthorized: '
    'Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses'
    ', cf-ray: a39429ffb948efdd-ICN, request id:'
    ' req_06cff9b298c34069a22c9a7d6837a5b4)"}',
    '{"type":"turn.failed","error":{"message":"unexpected status 401 Unauthorized: Missing '
    'bearer or basic authentication in header, url: https://api.openai.com/v1/responses,'
    ' cf-ray: a3942a15ebeaea1f-ICN, request id: req_c9f1f78058f845aba834462b6d971fa1"}}',
]
UNAUTHENTICATED_JSONL = "\n".join(UNAUTHENTICATED_LINES) + "\n"

# A future codex release adding item/event kinds this integration has never
# seen — synthetic, not captured, used only to test forward-compat.
FUTURE_COMPAT_LINES = [
    '{"type":"thread.started","thread_id":"th_future"}',
    '{"type":"turn.started"}',
    '{"type":"some_future_top_level_event_xyz","payload":{"whatever":true}}',
    '{"type":"item.started","item":{"id":"item_0","type":"some_future_item_kind_xyz",'
    '"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_0","type":"some_future_item_kind_xyz",'
    '"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"agent_message",'
    '"text":"still works"}}',
    '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,'
    '"output_tokens":1,"reasoning_output_tokens":0}}',
]
FUTURE_COMPAT_JSONL = "\n".join(FUTURE_COMPAT_LINES) + "\n"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStdin:
    """Records writes/close so tests can assert the stdin-hang guard holds."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.closed = False

    def write(self, text: str) -> None:
        self.written.append(text)

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    """Stands in for subprocess.Popen — never spawns anything real.

    `.stdout` / `.stderr` are plain `io.StringIO`, so `for line in
    proc.stdout` behaves exactly like a real text-mode pipe. `.poll()`
    defaults to "already exited" (0), matching `codex exec`'s own behaviour
    of exiting right after turn.completed/turn.failed, so turn()'s finally
    cleanup (_terminate_process) is a no-op in the ordinary translator
    tests; pass alive=True to exercise live termination instead.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        alive: bool = False,
        **kwargs: object,
    ) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.stdin = _FakeStdin()
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.terminate_calls = 0
        self.kill_calls = 0
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


class _FakeBridge:
    """A trivial `self._bridge` stand-in — pre-seeded directly, bypassing
    `_ensure_bridge()` so these tests need no real `imajin.agent.mcp_bridge`.
    """

    def __init__(self, url: str = "http://127.0.0.1:1/mcp", token: str = "fake-token") -> None:
        self.url = url
        self.token = token
        self.port = 1
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _dummy_caller(name: str, **kwargs: object) -> None:
    return None


def _make_runner(**kwargs: object) -> CodexAgentRunner:
    kwargs.setdefault("model", "gpt-5.6-sol")
    kwargs.setdefault("system_prompt", "be helpful")
    runner = CodexAgentRunner(**kwargs)
    runner._bridge = _FakeBridge()
    return runner


def _patch_codex(monkeypatch, stdout_texts: list[str], *, alive: bool = False) -> list[_FakePopen]:
    """Monkeypatch shutil.which + subprocess.Popen for one or more turns.

    Each call to the patched Popen pops the next entry of `stdout_texts` as
    that process's stdout. Returns the list of `_FakePopen` created, in call
    order, for inspecting argv/env/stdin per turn.
    """
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(
        codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH if name == "codex" else None
    )
    texts = list(stdout_texts)
    calls: list[_FakePopen] = []

    def factory(argv, **kwargs):
        text = texts.pop(0) if texts else ""
        proc = _FakePopen(argv, stdout_text=text, alive=alive, **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr(codex_agent.subprocess, "Popen", factory)
    return calls


@pytest.fixture(autouse=True)
def _stub_version_check(monkeypatch):
    """No test in this file exercises the version-guard's own behaviour
    (test_codex_version_is_cached_per_path uses the directly-imported
    original instead — see its comment); stub the module's own binding so
    turn() never shells out to a real `codex --version`, and so
    subprocess.Popen patches above can stay focused on simulating
    `codex exec` alone.
    """
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(codex_agent, "_codex_version", lambda path: (999, 0, 0))


# ---------------------------------------------------------------------------
# codex_available() — filesystem-only, never spawns, never reads contents
# ---------------------------------------------------------------------------


def test_codex_available_reports_missing_when_which_fails(monkeypatch):
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(codex_agent.shutil, "which", lambda name: None)
    assert codex_available() == (False, "codex not found")


def test_codex_available_reports_not_logged_in_when_auth_json_absent(tmp_path, monkeypatch):
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))  # empty dir, no auth.json
    assert codex_available() == (False, "not logged in")


def test_codex_available_happy_path(tmp_path, monkeypatch):
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text('{"tokens":{"access_token":"should-never-be-read"}}')
    assert codex_available() == (True, None)


def test_codex_available_never_reads_auth_json_contents(tmp_path, monkeypatch):
    """auth.json is a DIRECTORY, not a file: codex_available() must still say
    (True, None) from `.exists()` alone. Any attempt to open or read it as a
    file (read_text/read_bytes/open) would raise IsADirectoryError and fail
    this test — a self-contained proof that only existence is ever checked,
    with no need to monkeypatch pathlib internals globally.
    """
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").mkdir()
    assert codex_available() == (True, None)


# ---------------------------------------------------------------------------
# Pure translation helpers
# ---------------------------------------------------------------------------


def test_parse_version_handles_patch_and_no_patch():
    assert _parse_version("codex-cli 0.144.3") == (0, 144, 3)
    assert _parse_version("codex 1.2") == (1, 2)
    assert _parse_version("garbage, no version here") is None


def test_codex_version_is_cached_per_path(monkeypatch):
    """Uses the module's REAL _codex_version (imported directly above, so the
    autouse stub — which only rebinds codex_agent's own attribute — doesn't
    apply to this name) to verify the process is spawned at most once.
    """
    import imajin.agent.providers.codex_agent as codex_agent

    codex_agent._version_cache.clear()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout="codex-cli 0.144.3\n", stderr="")

    monkeypatch.setattr(codex_agent.subprocess, "run", fake_run)
    first = _codex_version(_FAKE_CODEX_PATH)
    second = _codex_version(_FAKE_CODEX_PATH)
    assert first == (0, 144, 3)
    assert second == (0, 144, 3)
    assert len(calls) == 1  # cached: the second call did not re-spawn


def test_flatten_mcp_result_variants():
    assert _flatten_mcp_result(None) == ""
    assert _flatten_mcp_result({"content": [{"type": "text", "text": "HELLO"}]}) == "HELLO"
    # non-text blocks are JSON-encoded rather than dropped
    out = _flatten_mcp_result({"content": [{"type": "image", "data": "x"}]})
    assert "image" in out


def test_tool_call_outcome_keys_on_status_not_on_error_field():
    """The exact trap the probe evidence names: a raised Imajin ToolError has
    status="failed" and error=null, with the message inside
    result.content[0].text. Keying on `error is not None` would call this a
    success.
    """
    raised = json.loads(ERRORPATH_LINES[4])["item"]
    assert raised["error"] is None  # the shape that makes the naive check wrong
    text, is_error = _tool_call_outcome(raised)
    assert is_error is True
    assert text == "imajin tool blew up: no image loaded"

    refused = json.loads(NOAPPROVAL_LINES[4])["item"]
    text, is_error = _tool_call_outcome(refused)
    assert is_error is True
    assert text == "user cancelled MCP tool call"

    ok = json.loads(APPROVE_LINES[4])["item"]
    text, is_error = _tool_call_outcome(ok)
    assert is_error is False
    assert text == "HELLO"


def test_translate_event_item_started_mcp_tool_call():
    event = json.loads(APPROVE_LINES[3])
    events = _translate_event(event, set())
    assert isinstance(events[0], ToolUseStart)
    assert events[0].id == "item_1" and events[0].name == "imajin_ping"
    assert isinstance(events[1], ToolUse)
    assert events[1].input == {"word": "hello"}


def test_translate_event_item_completed_agent_message():
    event = json.loads(APPROVE_LINES[5])
    (delta,) = _translate_event(event, set())
    assert isinstance(delta, TextDelta) and delta.text == "HELLO"


def test_translate_event_defensively_starts_a_tool_call_never_seen_started():
    """If a future codex version ever collapses started+completed into one
    event, the UI must still get a matching ToolUseStart/ToolUse before the
    ToolResult, not a dangling result."""
    event = json.loads(APPROVE_LINES[4])  # item.completed, never preceded by .started here
    events = _translate_event(event, set())
    assert [type(e).__name__ for e in events] == ["ToolUseStart", "ToolUse", "ToolResult"]


def test_translate_event_ignores_events_without_item():
    assert _translate_event({"type": "turn.started"}, set()) == []
    assert _translate_event({"type": "error", "message": "retry"}, set()) == []


def test_translate_event_ignores_unknown_item_type():
    event = {"item": {"id": "x", "type": "some_future_kind_xyz", "status": "completed"}}
    assert _translate_event(event, set()) == []


def test_looks_unauthenticated_matches_401_and_unauthorized():
    assert _looks_unauthenticated("unexpected status 401 Unauthorized: Missing bearer") is True
    assert _looks_unauthenticated("some other failure") is False


def test_failure_message_maps_401_to_a_clean_actionable_message():
    raw = json.loads(UNAUTHENTICATED_LINES[-1])["error"]
    assert _failure_message(raw) == "Sign in to Codex: run `codex login` in a terminal."


def test_failure_message_passes_other_errors_through():
    assert _failure_message({"message": "boom, something else"}) == "boom, something else"
    assert _failure_message(None) == "codex reported a turn failure with no message."


def test_terminate_process_escalates_to_kill_only_after_a_timeout():
    class _StubbornProc:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self._waits = 0

        def poll(self):
            return None  # never reaped via poll() in this test

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)
            return 0  # the second wait (after kill()) succeeds

        def kill(self):
            self.kill_calls += 1

    proc = _StubbornProc()
    _terminate_process(proc, timeout=0.01)
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1


# ---------------------------------------------------------------------------
# CodexAgentRunner lifecycle
# ---------------------------------------------------------------------------


def test_runner_lifecycle_flags():
    runner = CodexAgentRunner(model="gpt-5.6-sol", system_prompt="be helpful")
    assert runner.model == "gpt-5.6-sol"
    assert runner.name == "codex-agent"
    assert runner.max_turns == 24
    runner._thread_id = "th_1"
    runner.cancel()
    assert runner._cancelled is True
    runner.reset()
    assert runner._cancelled is False
    assert runner._thread_id is None


def test_cancel_terminates_the_process_without_blocking():
    """cancel() may run on the Qt main thread (chat_dock's Stop button is a
    direct-connect signal), so it must send the signal and return — never
    wait or escalate to kill(). That's turn()'s own finally block's job, on
    the background worker thread.
    """
    runner = _make_runner()
    fake_proc = _FakePopen(["x"], alive=True)
    runner._proc = fake_proc
    runner.cancel()
    assert runner._cancelled is True
    assert fake_proc.terminate_calls == 1
    assert fake_proc.kill_calls == 0


def test_close_stops_the_bridge_and_terminates_a_live_process():
    runner = _make_runner()
    bridge = runner._bridge
    fake_proc = _FakePopen(["x"], alive=True)
    runner._proc = fake_proc
    runner.close()
    assert bridge.stop_calls == 1
    assert fake_proc.terminate_calls == 1
    assert runner._bridge is None


def test_reset_drops_thread_id_but_leaves_the_bridge_running():
    runner = _make_runner()
    bridge = runner._bridge
    runner._thread_id = "th_1"
    runner._cancelled = True
    runner.reset()
    assert runner._thread_id is None
    assert runner._cancelled is False
    assert runner._bridge is bridge  # only close() stops it


def test_resolve_tool_caller_defaults_to_call_tool():
    from imajin.tools import call_tool

    runner = CodexAgentRunner(model="m", system_prompt="s")
    assert runner._resolve_tool_caller() is call_tool


def test_resolve_tool_caller_prefers_the_injected_caller():
    runner = CodexAgentRunner(model="m", system_prompt="s", tool_caller=_dummy_caller)
    assert runner._resolve_tool_caller() is _dummy_caller


def test_tool_names_selects_top_level_llm_tools_only():
    import imajin.tools  # noqa: F401 - populate the registry
    from imajin.tools.registry import iter_tools

    runner = CodexAgentRunner(model="m", system_prompt="s")
    names = runner._tool_names()
    expected = {e.name for e in iter_tools() if e.subagent is None and e.llm}
    assert names and set(names) == expected


def test_bridge_cls_resolves_to_the_real_imajin_mcp_bridge():
    """Confirms the lazy loader's import path is correct against the actual
    sibling-slice module — no socket bound, no thread started, just the class
    object.
    """
    from imajin.agent.mcp_bridge import ImajinMcpBridge
    from imajin.agent.providers.codex_agent import _bridge_cls

    assert _bridge_cls() is ImajinMcpBridge


def test_ensure_bridge_starts_once_against_the_pinned_constructor_shape(monkeypatch):
    """Verifies _ensure_bridge() calls the bridge class the way the pinned
    ImajinMcpBridge.__init__(tool_caller, *, tool_names=None) is declared —
    catches a contract mismatch even before the real bridge module exists.
    """
    import imajin.tools  # noqa: F401 - populate the registry
    import imajin.agent.providers.codex_agent as codex_agent

    created: list[_FakeBridge] = []

    class _TrackedFakeBridge(_FakeBridge):
        def __init__(self, tool_caller, *, tool_names=None):
            super().__init__()
            self.tool_caller = tool_caller
            self.tool_names = tool_names
            created.append(self)

    monkeypatch.setattr(codex_agent, "_bridge_cls", lambda: _TrackedFakeBridge)

    runner = CodexAgentRunner(model="m", system_prompt="s")
    first = runner._ensure_bridge()
    second = runner._ensure_bridge()

    assert first is second  # started once, reused
    assert len(created) == 1
    assert created[0].start_calls == 1
    assert created[0].tool_names  # non-empty: top-level LLM tools were passed
    assert callable(created[0].tool_caller)


# ---------------------------------------------------------------------------
# turn() — process plumbing (argv, stdin, env) and end-to-end translation
# ---------------------------------------------------------------------------


def test_stdin_is_written_and_closed(monkeypatch):
    """Regression guard for the confirmed indefinite hang: an open stdin pipe
    produced 18.2s and ZERO stdout lines, not even thread.started."""
    calls = _patch_codex(monkeypatch, [APPROVE_JSONL])
    runner = _make_runner(system_prompt="SYSTEM PROMPT HERE")
    list(runner.turn("hello codex"))

    assert len(calls) == 1
    assert calls[0].stdin.written == ["SYSTEM PROMPT HERE\n\nhello codex"]
    assert calls[0].stdin.closed is True


def test_every_popen_uses_the_resolved_absolute_path_never_the_bare_name(monkeypatch):
    calls = _patch_codex(monkeypatch, [APPROVE_JSONL, RESUME_JSONL])
    runner = _make_runner()
    list(runner.turn("one"))
    list(runner.turn("two"))
    assert calls[0].argv[0] == _FAKE_CODEX_PATH
    assert calls[1].argv[0] == _FAKE_CODEX_PATH
    assert calls[0].argv[0] != "codex"
    assert calls[1].argv[0] != "codex"


def test_bearer_token_never_appears_in_argv_only_in_env(monkeypatch):
    calls = _patch_codex(monkeypatch, [APPROVE_JSONL])
    runner = _make_runner()
    runner._bridge = _FakeBridge(token="s3cr3t-token-value")
    list(runner.turn("hi"))

    proc = calls[0]
    assert not any("s3cr3t-token-value" in arg for arg in proc.argv)
    assert proc.kwargs["env"]["IMAJIN_MCP_TOKEN"] == "s3cr3t-token-value"
    # argv names only the env var, never the secret value
    assert any("bearer_token_env_var" in arg for arg in proc.argv)
    assert any("default_tools_approval_mode" in arg for arg in proc.argv)


def test_thread_id_captured_and_used_on_resume(monkeypatch):
    calls = _patch_codex(monkeypatch, [APPROVE_JSONL, RESUME_JSONL])
    runner = _make_runner()

    list(runner.turn("turn one"))
    assert "resume" not in calls[0].argv
    assert runner._thread_id == "01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"

    list(runner.turn("turn two"))
    assert "resume" in calls[1].argv
    idx = calls[1].argv.index("resume")
    assert calls[1].argv[idx + 1] == "01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"
    assert calls[1].argv[idx + 2] == "-"  # still reads the prompt from stdin


def test_turn_translates_the_approve_fixture_end_to_end(monkeypatch):
    _patch_codex(monkeypatch, [APPROVE_JSONL])
    runner = _make_runner()
    events = list(runner.turn("please ping"))

    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert texts == ["I’ll call the requested tool directly.", "HELLO"]

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    uses = [e for e in events if isinstance(e, ToolUse)]
    results = [e for e in events if isinstance(e, ToolResult)]
    assert [s.name for s in starts] == ["imajin_ping"]
    assert uses[0].input == {"word": "hello"}
    assert results[0].is_error is False
    assert results[0].output == "HELLO"

    assert isinstance(events[-1], TurnDone)
    assert events[-1].stop_reason == "end_turn"
    # cache_read_input_tokens is the alias _map_usage adds so the chat dock's
    # usage line -- which reads that key, as the other three backends emit it --
    # shows the real cached count instead of a hardcoded-looking 0.
    assert events[-1].total_usage == {
        "input_tokens": 47759,
        "cached_input_tokens": 36992,
        "cache_read_input_tokens": 36992,
        "output_tokens": 136,
        "reasoning_output_tokens": 27,
    }
    assert runner._thread_id == "01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"


def test_turn_surfaces_the_noapproval_tool_refusal_as_a_failure(monkeypatch):
    """Regression guard: codex silently AUTO-CANCELS every MCP tool call when
    default_tools_approval_mode isn't "approve" — turn.completed still
    fires, exit 0, stderr empty (see codex_agent.py's module docstring / the
    probe evidence). If this translator ever regressed to only emitting a
    ToolResult on status=="completed" (silently dropping failed calls), this
    fixture is what would catch it: the refused call must still surface as
    an errored ToolResult even though the outer turn "completed" fine.
    """
    _patch_codex(monkeypatch, [NOAPPROVAL_JSONL])
    runner = _make_runner()
    events = list(runner.turn("please ping"))

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].output == "user cancelled MCP tool call"
    # codex's own signal says the turn completed; TurnDone reports that
    # faithfully — the failure surfaces at the tool-call level instead.
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "end_turn"


def test_turn_reports_two_tool_calls_one_raised_one_ok(monkeypatch):
    _patch_codex(monkeypatch, [ERRORPATH_JSONL])
    runner = _make_runner()
    events = list(runner.turn("ping boom then ok"))

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 2
    assert results[0].is_error is True
    assert results[0].output == "imajin tool blew up: no image loaded"
    assert results[1].is_error is False
    assert results[1].output == "OK"
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "end_turn"


def test_turn_maps_the_unauthenticated_failure_to_a_clean_message(monkeypatch):
    _patch_codex(monkeypatch, [UNAUTHENTICATED_JSONL])
    runner = _make_runner()
    events = list(runner.turn("hello"))

    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert any("codex login" in t for t in deltas)
    assert not any("401" in t for t in deltas)  # the raw status text must not leak through
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "error"
    # bare mid-stream {"type":"error",...} notices (10 of them, real capture)
    # must not have ended the turn early or produced their own TextDeltas.
    assert len(deltas) == 1


def test_turn_reports_an_unexpected_internal_error_gracefully(monkeypatch):
    """Safety net matching claude_agent.py's own pattern: nothing in the
    turn-driving block may propagate as a raw exception out of turn() — it
    must degrade to a TextDelta + an error TurnDone instead. Simulated here
    via a stdin.write() that raises (e.g. a BrokenPipeError if codex's
    process already exited right after spawning).
    """
    import imajin.agent.providers.codex_agent as codex_agent

    class _BoomStdin(_FakeStdin):
        def write(self, text: str) -> None:
            raise BrokenPipeError("pipe closed by peer")

    created: list[_FakePopen] = []

    def factory(argv, **kwargs):
        proc = _FakePopen(argv, stdout_text=APPROVE_JSONL, **kwargs)
        proc.stdin = _BoomStdin()
        created.append(proc)
        return proc

    monkeypatch.setattr(
        codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH if name == "codex" else None
    )
    monkeypatch.setattr(codex_agent.subprocess, "Popen", factory)

    runner = _make_runner()
    events = list(runner.turn("hi"))

    assert created[0].stdin.closed is True  # closed despite the raised write
    assert any(isinstance(e, TextDelta) and "BrokenPipeError" in e.text for e in events)
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "error"


def test_unknown_item_and_top_level_types_do_not_crash(monkeypatch):
    _patch_codex(monkeypatch, [FUTURE_COMPAT_JSONL])
    runner = _make_runner()
    events = list(runner.turn("hi"))

    assert any(isinstance(e, TextDelta) and e.text == "still works" for e in events)
    assert isinstance(events[-1], TurnDone)
    assert events[-1].stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# A refused `codex exec resume` must not wedge the session
# ---------------------------------------------------------------------------

# Verbatim shape of a refused resume, captured against real codex 0.144.3 (zero
# quota — it fails before any model call): PLAIN TEXT on stderr, exit code 0,
# and ZERO JSONL on stdout, not even thread.started. codex prunes its own
# rollout files, so this is reachable in normal use, not a corner case.
RESUME_REFUSED_STDERR = (
    "Error: thread/resume: thread/resume failed: no rollout found for thread id "
    "01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0 (code -32600)\n"
)


def test_a_refused_resume_drops_the_thread_id_instead_of_wedging_the_session(monkeypatch):
    """Turn 2's resume is refused; turn 3 must start a FRESH thread.

    Without dropping the id, every later turn resumes the same missing thread
    and fails identically — the chat is stuck until the user happens to press
    Clear, with nothing telling them that's the fix.
    """
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(
        codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH if name == "codex" else None
    )
    stdouts = [APPROVE_JSONL, "", APPROVE_JSONL]
    stderrs = ["", RESUME_REFUSED_STDERR, ""]
    calls: list[_FakePopen] = []

    def factory(argv, **kwargs):
        proc = _FakePopen(argv, stdout_text=stdouts.pop(0), stderr_text=stderrs.pop(0), **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr(codex_agent.subprocess, "Popen", factory)
    runner = _make_runner()

    list(runner.turn("turn one"))
    assert runner._thread_id == "01a08ee3-085c-71a0-ad4d-1b7b12b0c5b0"

    events = list(runner.turn("turn two"))
    # The user is told what codex said, rather than getting a silent dead turn.
    assert "no rollout found" in "".join(
        e.text for e in events if isinstance(e, TextDelta)
    )
    assert events[-1].stop_reason == "error"
    assert runner._thread_id is None

    list(runner.turn("turn three"))
    assert "resume" not in calls[2].argv, "turn 3 must start a fresh codex thread"


def test_stderr_reader_is_started_before_the_stdin_write(monkeypatch):
    """Ordering guard, asserted on the actual sequence of calls.

    codex logs to stderr while starting up, BEFORE it has drained stdin (its
    "could not create PATH aliases" warning is emitted that early). If nobody
    were reading stderr, a chatty startup that filled the 64 KB pipe buffer
    would block codex mid-write while we block mid-write to stdin with a system
    prompt that is already 34 KB and only grows -- a deadlock with no timeout on
    either side. So the drain thread must be running before the first byte of
    stdin goes out; this asserts that order directly rather than the weaker
    "stderr got read eventually".
    """
    import imajin.agent.providers.codex_agent as codex_agent

    monkeypatch.setattr(
        codex_agent.shutil, "which", lambda name: _FAKE_CODEX_PATH if name == "codex" else None
    )
    order: list[str] = []

    class _LoggingStdin(_FakeStdin):
        def write(self, text: str) -> None:
            order.append("stdin.write")
            super().write(text)

    real_thread = codex_agent.threading.Thread

    class _LoggingThread(real_thread):  # type: ignore[misc,valid-type]
        def start(self) -> None:
            order.append(f"thread.start:{self.name}")
            super().start()

    monkeypatch.setattr(codex_agent.threading, "Thread", _LoggingThread)

    def factory(argv, **kwargs):
        p = _FakePopen(argv, stdout_text=APPROVE_JSONL, stderr_text="warning: hello\n", **kwargs)
        p.stdin = _LoggingStdin()
        return p

    monkeypatch.setattr(codex_agent.subprocess, "Popen", factory)
    list(_make_runner().turn("hi"))

    assert order == ["thread.start:codex-agent-stderr", "stdin.write"], order


# ---------------------------------------------------------------------------
# max_turns — enforced here because codex exec has no turn-cap flag
# ---------------------------------------------------------------------------


def _mcp_call_line(item_id: str) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": item_id,
                "type": "mcp_tool_call",
                "server": "imajin",
                "tool": "list_layers",
                "arguments": {},
                "result": {"content": [{"type": "text", "text": "[]"}]},
                "error": None,
                "status": "completed",
            },
        }
    )


def test_turn_stops_once_max_turns_tool_calls_have_started(monkeypatch):
    """An uncapped loop matters more on this backend, not less: every MCP call
    is auto-approved with no human, and most bridged tools mutate the session.
    """
    lines = [json.dumps({"type": "thread.started", "thread_id": "th_cap"})]
    lines += [_mcp_call_line(f"item_{i}") for i in range(10)]
    lines.append(json.dumps({"type": "turn.completed", "usage": {}}))
    procs = _patch_codex(monkeypatch, ["\n".join(lines) + "\n"], alive=True)

    runner = _make_runner(max_turns=3)
    events = list(runner.turn("go"))

    done = [e for e in events if isinstance(e, TurnDone)]
    assert len(done) == 1 and done[0].stop_reason == "max_turns"
    # Capped at max_turns + 1 started calls: the cap trips *after* the call that
    # crossed it is translated, so the model's last action is still reported.
    tool_uses = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_uses) == 4
    assert procs[0].terminate_calls or procs[0].kill_calls
    # The thread id survives, so the user can narrow the request and continue.
    assert runner._thread_id == "th_cap"


def test_turn_under_the_cap_completes_normally(monkeypatch):
    lines = [json.dumps({"type": "thread.started", "thread_id": "th_ok"})]
    lines += [_mcp_call_line(f"item_{i}") for i in range(2)]
    lines.append(json.dumps({"type": "turn.completed", "usage": {}}))
    _patch_codex(monkeypatch, ["\n".join(lines) + "\n"])

    runner = _make_runner(max_turns=24)
    done = [e for e in runner.turn("go") if isinstance(e, TurnDone)]
    assert len(done) == 1 and done[0].stop_reason == "end_turn"


def test_web_search_is_disabled_in_the_argv(monkeypatch):
    """Parity with WebSearch/WebFetch in claude_agent._DISALLOWED_BUILTINS."""
    procs = _patch_codex(monkeypatch, [json.dumps({"type": "turn.completed", "usage": {}}) + "\n"])
    list(_make_runner().turn("hi"))
    assert "tools.web_search=false" in procs[0].argv


def test_map_usage_normalises_the_cache_key_to_the_house_convention():
    """chat_dock's usage line reads cache_read_input_tokens, which the other
    three backends all emit; codex calls it cached_input_tokens. Without the
    alias every Codex turn displayed "cache_read 0" against a real 42,752.
    """
    from imajin.agent.providers.codex_agent import _map_usage

    usage = _map_usage({"input_tokens": 48893, "cached_input_tokens": 42752, "output_tokens": 111})
    assert usage["cache_read_input_tokens"] == 42752
    assert usage["cached_input_tokens"] == 42752  # codex's own key is kept too
    assert _map_usage({"input_tokens": 10}) == {"input_tokens": 10}
