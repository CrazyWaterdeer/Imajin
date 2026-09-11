"""Subscription-backed agent that runs on the user's own logged-in `codex` CLI.

Like :class:`~imajin.agent.providers.claude_agent.ClaudeAgentRunner`, this is
**not** a :class:`~imajin.agent.providers.base.Provider` — codex owns its own
agentic tool-calling loop (it drives the `codex` CLI, which executes tools
itself), so it cannot sit behind ``AgentRunner``'s ``stream()``. Instead this
class fuses provider + runner: it presents the same ``turn()`` / ``reset()`` /
``cancel()`` / ``close()`` surface the chat dock already drives, while codex's
own loop does the actual work. Imajin's own tools are bridged in as an
in-process, loopback-only MCP server (:class:`~imajin.agent.mcp_bridge.ImajinMcpBridge`)
so codex's loop calls back into this process against the *live* napari
session; codex's JSONL event stream is translated back into Imajin
``RunEvent`` objects for the UI.

Structurally this is **simpler** than ``ClaudeAgentRunner``: ``codex exec`` is
one-shot per turn with a real OS process handle (Popen), not a persistent SDK
connection — so there is no asyncio loop/thread/client to keep alive across
turns. What *does* persist across turns on one runner instance is the MCP
bridge (started once, reused — see ``_ensure_bridge``) and the codex-side
thread id (``codex exec resume <id>``).

Auth: the `codex` CLI resolves its own credentials from `codex login`, so no
API key is needed — it uses whatever ChatGPT/subscription login the user set
up themselves in a terminal. This module ships the **unmodified** codex CLI,
builds no login UI, and never reads, parses, stores, or forwards the contents
of ``~/.codex/auth.json`` — only its *existence* is checked, in
:func:`codex_available`. This mirrors the decision already recorded for
Claude in ``claude_agent.py``'s module docstring, and for the same reason:
reading the token and calling an API directly would violate the ChatGPT
Consumer Terms, and is technically dead anyway — ChatGPT-auth codex talks to
``chatgpt.com/backend-api/codex/responses``, not ``api.openai.com``.

Security — where this backend is DELIBERATELY weaker than the Claude one, and
why. ``claude_agent.py`` denies a fixed list of built-ins
(``_DISALLOWED_BUILTINS``), sets ``permission_mode="dontAsk"`` and
``setting_sources=[]`` so ambient ``~/.claude`` and project settings are
ignored. codex has no equivalent for most of that, so three gaps stand:

* **Shell.** ``approval_policy=never`` + ``sandbox_mode=read-only`` means codex
  auto-runs *read-only* shell commands anywhere on the user's filesystem
  without asking. It cannot write, but it can read. There is no flag to remove
  the shell tool from ``codex exec``.
* **Ambient config.** ``codex exec`` layers our ``-c`` overrides on top of the
  user's own ``~/.codex/config.toml``, so any other MCP servers they configured
  are also loaded and connected. ``CODEX_HOME`` is the only isolation lever and
  repointing it would hide ``auth.json`` too, breaking the login — so this is
  accepted, not fixed.
* **Tool approval.** ``default_tools_approval_mode="approve"`` is load-bearing
  (without it every MCP call is silently auto-cancelled while codex still
  reports success), and it auto-approves every call with no human in the loop.
  Most bridged Imajin tools mutate the live napari session, so the only bound
  on damage is ``max_turns``, enforced by hand in :meth:`CodexAgentRunner.turn`.

``tools.web_search=false`` closes the one gap that *was* cheaply closable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from imajin.agent.providers.base import TextDelta, ToolUse, ToolUseStart
from imajin.agent.runner import ToolResult, TurnDone

# Bridged Imajin tools live under this MCP server name; codex sees each tool
# as `<name>__<tool>` (the exact display form is codex's concern, not ours —
# we only need the name for the `-c mcp_servers.<name>.*` config keys).
_MCP_SERVER = "imajin"

# The bridge's per-session bearer token goes to codex through this env var
# name, never inlined in a `-c` value: `-c` values land in the child's argv,
# which is readable by any local user via /proc (Linux) or Task Manager
# (Windows) — the same lesson claude_agent.py already records for its own
# subprocess-adjacent env handling. `bearer_token_env_var` is also the only
# option: codex rejects `mcp_servers.<n>.bearer_token` outright for
# streamable_http ("bearer_token is not supported for streamable_http").
_TOKEN_ENV_VAR = "IMAJIN_MCP_TOKEN"

# codex's own auth precedence puts an API key ahead of the ChatGPT
# subscription login (codex-rs/login/src/auth/manager.rs: "API key via env
# var takes precedence over any other auth method"), so a stray key in the
# parent environment would silently bill the API instead of using the
# subscription. Stripped from the child env for every turn — mirrors
# claude_agent.py's _AUTH_ENV_KEYS / _force_subscription_env for
# ANTHROPIC_API_KEY, though here it's cheap enough to do unconditionally
# rather than needing a save/restore around a shared process environment.
_AUTH_ENV_KEYS = ("CODEX_API_KEY", "CODEX_ACCESS_TOKEN")

# Keep tool-calling turns fast/cheap. Not exposed as a constructor param (the
# pinned CodexAgentRunner signature has none) — only "low" was verified.
_MODEL_REASONING_EFFORT = "low"

# codex's event schema and `-c` config-key names are version-specific (this
# integration was probed against 0.144.3); warn, don't fail, below this.
_MIN_VERSION = (0, 144)

_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_version_cache: dict[str, tuple[int, ...] | None] = {}
_version_cache_lock = threading.Lock()


def codex_available() -> tuple[bool, str | None]:
    """Whether the Codex subscription path is usable, and a short reason if not.

    Filesystem-only and **never spawns a subprocess**: compute_statuses runs
    synchronously on the Qt main thread at ChatDock construction and on every
    settings save (see ``subscription_available()``, ``ClaudeAgentRunner``'s
    sibling probe, for the same constraint), so a hung `codex` binary here
    would freeze the UI. auth.json is checked for **existence only** — this
    function must never open, read, or parse it; see this module's docstring
    for why.
    """
    if shutil.which("codex") is None:
        return False, "codex not found"
    codex_home = os.environ.get("CODEX_HOME")
    auth_path = (Path(codex_home) if codex_home else Path.home() / ".codex") / "auth.json"
    if auth_path.exists():
        return True, None
    return False, "not logged in"


def _parse_version(text: str) -> tuple[int, ...] | None:
    """Parse a `codex --version` banner such as "codex-cli 0.144.3" into (0, 144, 3)."""
    match = _VERSION_RE.search(text)
    if not match:
        return None
    return tuple(int(g) for g in match.groups() if g is not None)


def _codex_version(codex_path: str) -> tuple[int, ...] | None:
    """`codex --version`, parsed and cached per resolved path.

    Spawns a process, so this must stay off any UI-blocking hot path —
    ``codex_available()`` above deliberately does not call it.
    ``CodexAgentRunner`` calls it lazily from ``turn()`` rather than
    ``__init__``, so constructing a runner never spawns anything either.
    """
    with _version_cache_lock:
        if codex_path in _version_cache:
            return _version_cache[codex_path]
    version: tuple[int, ...] | None = None
    try:
        result = subprocess.run(
            [codex_path, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = _parse_version(result.stdout or result.stderr)
    except (OSError, subprocess.TimeoutExpired):
        version = None
    with _version_cache_lock:
        _version_cache[codex_path] = version
    return version


def _bridge_cls() -> Any:
    """Import the in-process MCP bridge lazily.

    Deferred the same way ``claude_agent.py``'s ``_sdk()`` defers
    ``claude_agent_sdk``: it keeps this module importable while
    ``imajin.agent.mcp_bridge`` (a sibling slice) is mid-flight, and lets
    tests substitute a fake bridge without a real one ever binding a socket.
    """
    from imajin.agent.mcp_bridge import ImajinMcpBridge  # noqa: PLC0415

    return ImajinMcpBridge


def _map_usage(raw: Any) -> dict[str, int]:
    """codex's ``turn.completed`` usage dict, filtered to the known int fields.

    Exactly these four keys are documented from live captures (no cost, no
    totals). ``input_tokens`` is the *whole resent context* each turn, not a
    delta — it grows turn over turn and must never be summed as if it were.
    """
    if not isinstance(raw, dict):
        return {}
    out = {k: v for k, v in raw.items() if k in _USAGE_KEYS and isinstance(v, int)}
    # codex calls it cached_input_tokens; anthropic.py, openai_compat.py and
    # claude_agent.py all emit cache_read_input_tokens, which is the key the
    # chat dock's usage line reads. Without this the UI shows "cache_read 0" on
    # every Codex turn while the real number is most of the context.
    if "cached_input_tokens" in out:
        out.setdefault("cache_read_input_tokens", out["cached_input_tokens"])
    return out


def _flatten_mcp_result(result: Any) -> str:
    """Flatten an ``mcp_tool_call`` item's ``result`` (MCP CallToolResult shape,
    e.g. ``{"content":[{"type":"text","text":"HELLO"}],"structured_content":null}``)
    to display text. Non-text blocks are JSON-encoded rather than dropped, so
    nothing silently vanishes (mirrors ``claude_agent._flatten_tool_result``).
    """
    if result is None:
        return ""
    if not isinstance(result, dict):
        return str(result)
    content = result.get("content")
    if not isinstance(content, list):
        return json.dumps(result, default=str)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, dict):
            parts.append(json.dumps(block, default=str))
        else:
            parts.append(str(block))
    return "".join(parts)


def _tool_call_outcome(item: dict[str, Any]) -> tuple[str, bool]:
    """(display text, is_error) for a completed ``mcp_tool_call`` item.

    Keyed on ``status``, **not** on whether ``error`` is set: a raised Imajin
    ``ToolError`` comes back as ``status="failed"`` with ``error=null`` and the
    message inside ``result.content[0].text`` (codex_events_errorpath.jsonl);
    a codex-side refusal (an unapproved/cancelled tool call) sets
    ``error.message`` with ``result=null`` (codex_events_noapproval.jsonl).
    Reading only ``error`` marks the first shape — the common one, since
    Imajin tools raise ``ToolError`` routinely — as a SUCCESS.
    """
    is_error = item.get("status") == "failed"
    error = item.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message), is_error
    return _flatten_mcp_result(item.get("result")), is_error


def _translate_event(event: dict[str, Any], started_tool_calls: set[str]) -> list[Any]:
    """Translate one ``item.started`` / ``item.completed`` line into RunEvents.

    Permissive by construction: an unrecognized ``item.type`` — a future kind
    codex adds between patch releases, or one of reasoning / command_execution
    / file_change / collab_tool_call / web_search / todo_list / error that
    Imajin's chat dock has no rendering for yet — produces no events rather
    than raising. Terminal events (``turn.completed`` / ``turn.failed``),
    ``thread.started``, and the bare top-level ``{"type": "error", ...}`` log
    notice all carry no ``item``, so they safely fall through to "no events"
    here too; the caller handles them separately.

    ``started_tool_calls`` is turn-local: each :meth:`CodexAgentRunner.turn`
    call makes a fresh set for its one subprocess, so the "item ids reset to
    item_0 on a resumed turn" trap (codex_events_resume.jsonl restarts at
    item_0 inside the *same* thread_id) can't collide across turns here —
    there is no cross-turn dict to collide in.
    """
    events: list[Any] = []
    item = event.get("item")
    if not isinstance(item, dict):
        return events
    item_type = item.get("type")
    etype = event.get("type")

    if item_type == "mcp_tool_call":
        item_id = str(item.get("id", ""))
        name = str(item.get("tool") or "tool")
        if etype == "item.started":
            started_tool_calls.add(item_id)
            events.append(ToolUseStart(id=item_id, name=name))
            events.append(ToolUse(id=item_id, name=name, input=dict(item.get("arguments") or {})))
        elif etype == "item.completed":
            if item_id not in started_tool_calls:
                # Defensive: every observed run had item.started first. Don't
                # hand the UI a ToolResult with no matching ToolUseStart if a
                # future codex version ever collapses the two into one event.
                started_tool_calls.add(item_id)
                events.append(ToolUseStart(id=item_id, name=name))
                events.append(
                    ToolUse(id=item_id, name=name, input=dict(item.get("arguments") or {}))
                )
            output, is_error = _tool_call_outcome(item)
            events.append(
                ToolResult(tool_use_id=item_id, name=name, output=output, is_error=is_error)
            )
        return events

    if item_type == "agent_message" and etype == "item.completed":
        # No item.started and no item.updated for agent_message in any captured
        # run — text arrives whole, so one TextDelta per completed message.
        text = item.get("text")
        if text:
            events.append(TextDelta(text=str(text)))
        return events

    return events


def _looks_unauthenticated(message: str) -> bool:
    lowered = message.lower()
    return "401" in lowered and "unauthorized" in lowered


def _failure_message(error: Any) -> str:
    """``turn.failed``'s display text, mapped to a clean, actionable message for
    the one failure mode expected to be common (a stale/missing login) rather
    than a raw 401; anything else is passed through as codex reported it.
    """
    raw = ""
    if isinstance(error, dict):
        raw = str(error.get("message") or "")
    if _looks_unauthenticated(raw):
        return "Sign in to Codex: run `codex login` in a terminal."
    return raw or "codex reported a turn failure with no message."


def _drain_stream(stream: Any, sink: list[str], *, max_lines: int = 200) -> None:
    """Read a subprocess pipe to EOF on a dedicated thread.

    Reading stdout and stderr from the same thread risks a classic pipe
    deadlock: if stderr fills its OS buffer while we're blocked reading
    stdout (or vice versa), the child blocks writing and we block reading,
    forever. A thread per stream avoids it. Best-effort only — stderr is a
    log, not a protocol, so a read error here must not fail the turn.
    """
    try:
        for line in stream:
            if len(sink) < max_lines:
                sink.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass


def _terminate_process(proc: subprocess.Popen[str], *, timeout: float = 5.0) -> None:
    """SIGTERM, wait, and escalate to SIGKILL only if that doesn't work.

    The npm `codex` launcher is a Node shim that spawns the real Rust binary
    with ``stdio="inherit"`` and explicitly forwards SIGINT/SIGTERM/SIGHUP —
    but it cannot forward SIGKILL, since SIGKILL doesn't give the shim a
    chance to run any handler at all. Kill the shim outright (Popen.kill())
    and the Rust grandchild survives, holding the stdout pipe open, so a
    reader blocked on it never sees EOF (confirmed: kill_vs_term.py — SIGTERM
    reached EOF within 8s with 0 surviving grandchildren; SIGKILL hung
    ``communicate()`` outright). terminate() first, always; kill() is the
    last resort, not the first move.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


class CodexAgentRunner:
    """Runner backed by the user's own logged-in `codex` CLI subscription.

    Fused provider+runner like ``ClaudeAgentRunner`` (see that class's
    docstring for why): presents ``turn()`` / ``reset()`` / ``cancel()`` /
    ``close()`` yielding the shared ``RunEvent`` types, while codex owns its
    own agentic tool-calling loop internally. Structurally simpler, though:
    ``codex exec`` is a one-shot process per turn with a real OS process
    handle, so there is no persistent asyncio client/loop/thread to own —
    ``turn()`` just Popens, writes+closes stdin, and reads JSONL off stdout
    synchronously on the calling (worker) thread.

    Imajin's own tools are bridged in via
    :class:`imajin.agent.mcp_bridge.ImajinMcpBridge`, an in-process
    streamable-HTTP MCP server on loopback — started once, lazily, on the
    first turn, and reused across turns (see ``_ensure_bridge``). codex is
    pointed at it with per-run ``-c mcp_servers.<name>.*`` flags that need no
    ``~/.codex/config.toml`` edit.
    """

    name = "codex-agent"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tool_caller: Any | None = None,
        max_turns: int = 24,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        # codex `exec` has no flag of its own for capping its internal
        # tool-calling loop (0.144.3's config schema and --help were both
        # searched), so unlike ClaudeAgentRunner — which hands max_turns to the
        # SDK — this runner enforces the cap itself in turn(). See the comment
        # at the check for why it matters more on this backend, not less.
        self.max_turns = max_turns
        self._tool_caller = tool_caller
        self._cancelled = False
        self._thread_id: str | None = None
        self._bridge: Any | None = None
        self._proc: subprocess.Popen[str] | None = None
        # Fixed for the runner's whole lifetime: `codex exec resume` filters
        # candidate threads by cwd, so a cwd that moved between turns could
        # make an earlier turn's thread unresumable.
        self._cwd = tempfile.gettempdir()
        self._version_warned = False

    # -- lifecycle (mirrors ClaudeAgentRunner) --------------------------------

    def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None:
            # Fire-and-forget, like ClaudeAgentRunner.cancel(): this runs on
            # the Qt main thread (chat_dock's Stop button is a direct-connect
            # signal), so it must not block. Send the signal only — turn()'s
            # own `finally`, on the background worker thread, does the
            # wait/escalate-to-kill via _terminate_process.
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort; turn()'s cleanup still runs
                pass

    def reset(self) -> None:
        # Drop the resumable thread id so the next turn starts a fresh
        # conversation. Does NOT stop the bridge or kill an in-flight
        # process — cancel()/close() own those.
        self._cancelled = False
        self._thread_id = None

    def close(self) -> None:
        """Tear down any in-flight process and the MCP bridge.

        Call when discarding the runner. May block briefly (terminate → wait
        → kill, up to a few seconds) — the same tradeoff
        ClaudeAgentRunner.close() already makes with its ``thread.join()`` /
        ``disconnect()``; close() is teardown, not a per-turn hot path.
        """
        proc = self._proc
        if proc is not None:
            _terminate_process(proc)
        bridge, self._bridge = self._bridge, None
        if bridge is not None:
            try:
                bridge.stop()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    # -- tool bridge -----------------------------------------------------------

    def _resolve_tool_caller(self) -> Callable[..., Any]:
        if self._tool_caller is not None:
            return self._tool_caller
        from imajin.tools import call_tool  # noqa: PLC0415

        return call_tool

    def _tool_names(self) -> list[str]:
        from imajin.tools.registry import iter_tools  # noqa: PLC0415

        # Same selection as ClaudeAgentRunner._bridged_entries() /
        # tools_for_anthropic(): top-level LLM tools, no subagent-only
        # entries. Keeps parity across backends and keeps the MCP
        # `tools/list` payload down.
        return [e.name for e in iter_tools() if e.subagent is None and e.llm]

    def _ensure_bridge(self) -> Any:
        if self._bridge is not None:
            return self._bridge
        bridge_cls = _bridge_cls()
        bridge = bridge_cls(self._resolve_tool_caller(), tool_names=self._tool_names())
        bridge.start()
        self._bridge = bridge
        return bridge

    # -- process plumbing -------------------------------------------------------

    def _build_argv(self, codex_path: str, bridge_url: str) -> list[str]:
        """The verified working invocation (see this module's docstring),
        with the bearer token routed through an env var instead of a `-c`
        value — see :meth:`_build_env`.
        """
        argv = [
            codex_path,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{_MODEL_REASONING_EFFORT}"',
            "-c",
            "approval_policy=never",
            "-c",
            "sandbox_mode=read-only",
            "-c",
            f'mcp_servers.{_MCP_SERVER}.url="{bridge_url}"',
            "-c",
            f'mcp_servers.{_MCP_SERVER}.bearer_token_env_var="{_TOKEN_ENV_VAR}"',
            # Orthogonal to approval_policy and load-bearing on its own:
            # without this, every MCP tool call is silently auto-cancelled
            # (status="failed", exit 0, empty stderr —
            # codex_events_noapproval.jsonl) while codex still reports
            # turn.completed. approval_policy=never does not cover MCP tools
            # (codex-rs/codex-mcp/src/mcp/mod.rs's auto-approve check looks at
            # this flag before it ever reads approval_policy).
            "-c",
            f'mcp_servers.{_MCP_SERVER}.default_tools_approval_mode="approve"',
            # Parity with WebSearch/WebFetch in claude_agent._DISALLOWED_BUILTINS:
            # an embedded, napari-bound analysis agent has no business browsing.
            # Top-level key, not tools.web_search=<bool> under another name —
            # verified accepted by `codex exec --strict-config`.
            "-c",
            "tools.web_search=false",
        ]
        if self._thread_id is not None:
            # The subcommand and its args go AFTER the global -c flags.
            argv += ["resume", self._thread_id]
        argv.append("-")  # read the prompt from stdin
        return argv

    def _build_env(self, token: str) -> dict[str, str]:
        env = dict(os.environ)
        for key in _AUTH_ENV_KEYS:
            env.pop(key, None)
        env[_TOKEN_ENV_VAR] = token
        return env

    def _build_stdin_text(self, user_text: str) -> str:
        # codex has no separate system-prompt channel like Claude's
        # --system-prompt-file; the whole prompt goes over stdin as one blob.
        # Sent on every turn (not just the first) — this is what was actually
        # verified working; whether resumed turns could omit it and rely on
        # codex's own replayed thread history instead was not tested, so we
        # don't assume it.
        return f"{self.system_prompt}\n\n{user_text}"

    def _warn_if_outdated(self, codex_path: str) -> Iterator[Any]:
        """Emit a one-time warning if codex is older than the verified floor.

        Only checked once per runner instance — ``_codex_version()`` itself
        is cached per path at module scope, so this never respawns
        `codex --version` even across many runner instances.
        """
        if self._version_warned:
            return
        self._version_warned = True
        version = _codex_version(codex_path)
        if version is not None and version < _MIN_VERSION:
            have = ".".join(str(part) for part in version)
            need = ".".join(str(part) for part in _MIN_VERSION)
            yield TextDelta(
                text=(
                    f"[codex-agent] warning: codex {have} detected; this integration "
                    f"was verified against {need}+ and event parsing may be unreliable.\n"
                )
            )

    # -- turn driving ------------------------------------------------------------

    def turn(self, user_text: str) -> Iterator[Any]:
        """Drive one turn as one `codex exec` subprocess, yielding RunEvents.

        See this class's docstring for why there is no persistent connection
        to manage here, unlike ClaudeAgentRunner.
        """
        self._cancelled = False

        codex_path = shutil.which("codex")  # NEVER Popen a bare "codex"
        if codex_path is None:
            yield TextDelta(text="\n[codex agent error] codex CLI not found on PATH.")
            yield TurnDone(stop_reason="error", total_usage={})
            return

        try:
            bridge = self._ensure_bridge()
        except Exception as exc:  # noqa: BLE001 - report in-stream, don't crash the worker
            yield TextDelta(text=f"\n[codex agent error] {type(exc).__name__}: {exc}")
            yield TurnDone(stop_reason="error", total_usage={})
            return

        yield from self._warn_if_outdated(codex_path)

        argv = self._build_argv(codex_path, bridge.url)
        env = self._build_env(bridge.token)

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=env,
            )
        except OSError as exc:
            yield TextDelta(text=f"\n[codex agent error] failed to start codex: {exc}")
            yield TurnDone(stop_reason="error", total_usage={})
            return

        self._proc = proc
        stderr_lines: list[str] = []
        stderr_thread: threading.Thread | None = None
        saw_terminal = False
        unexpected_error: Exception | None = None
        try:
            # Started BEFORE the stdin write, not after: codex writes to stderr
            # while it is still starting up and has not yet drained stdin (e.g.
            # "WARNING: proceeding, even though we could not create PATH
            # aliases"). With nobody reading stderr, a chatty startup that fills
            # the 64 KB pipe buffer would block codex mid-write while we are
            # blocked writing a system prompt that is already 34 KB and only
            # grows — a deadlock neither side can break. Draining from the first
            # instant costs nothing and removes the race entirely.
            stderr_thread = threading.Thread(
                target=_drain_stream,
                args=(proc.stderr, stderr_lines),
                daemon=True,
                name="codex-agent-stderr",
            )
            stderr_thread.start()

            try:
                # KNOWN TRAP: an open stdin pipe hangs `codex exec` forever —
                # confirmed 18.2s with ZERO stdout lines, not even
                # thread.started (stdin_hang.py). Write and close in the same
                # breath; nothing between them may raise and skip the close.
                proc.stdin.write(self._build_stdin_text(user_text))
            finally:
                proc.stdin.close()

            started_tool_calls: set[str] = set()
            for raw_line in proc.stdout:
                if self._cancelled:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # stdout is documented pure JSONL; a stray line must not
                    # crash an otherwise-healthy turn.
                    continue
                if not isinstance(event, dict):
                    continue
                etype = event.get("type")

                if etype == "thread.started":
                    thread_id = event.get("thread_id")
                    if isinstance(thread_id, str) and thread_id:
                        self._thread_id = thread_id
                    continue

                if etype == "turn.completed":
                    yield TurnDone(
                        stop_reason="end_turn", total_usage=_map_usage(event.get("usage"))
                    )
                    saw_terminal = True
                    break

                if etype == "turn.failed":
                    yield TextDelta(
                        text=f"\n[codex agent error] {_failure_message(event.get('error'))}"
                    )
                    yield TurnDone(stop_reason="error", total_usage={})
                    saw_terminal = True
                    break

                # A bare {"type": "error", ...} is a non-fatal retry/log
                # notice (up to 10 appeared in one merely-reconnecting run) —
                # informational, never terminal on its own. turn.started and
                # any other/unknown top-level type fall through here too,
                # alongside it: _translate_event is a no-op for anything
                # without an "item", which covers all of them.
                yield from _translate_event(event, started_tool_calls)

                # The cap ClaudeAgentRunner gets from the SDK, enforced by hand.
                # It is MORE important here, not less: default_tools_approval_mode
                # ="approve" (load-bearing — without it every call is silently
                # auto-cancelled) auto-approves every MCP call with no human in
                # the loop, and most of the bridged Imajin tools mutate the live
                # napari session. An uncapped runaway would keep rewriting the
                # user's layers unopposed. The thread id is deliberately kept, so
                # the user can narrow the request and carry on in the same thread.
                if len(started_tool_calls) > self.max_turns:
                    _terminate_process(proc)
                    yield TextDelta(
                        text=(
                            f"\n[codex agent] stopped after {len(started_tool_calls)} tool "
                            f"calls (max_turns={self.max_turns}). Narrow the request and "
                            "send again to continue in the same conversation."
                        )
                    )
                    yield TurnDone(stop_reason="max_turns", total_usage={})
                    saw_terminal = True
                    break
        except Exception as exc:  # noqa: BLE001 - nothing in this block may crash the turn
            unexpected_error = exc
        finally:
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            _terminate_process(proc)  # no-op if it already exited on its own
            if stderr_thread is not None:
                stderr_thread.join(timeout=2)
            if self._proc is proc:
                self._proc = None

        if saw_terminal:
            return
        if unexpected_error is not None:
            yield TextDelta(
                text=f"\n[codex agent error] {type(unexpected_error).__name__}: {unexpected_error}"
            )
            yield TurnDone(stop_reason="error", total_usage={})
            return
        if self._cancelled:
            yield TurnDone(stop_reason="cancelled", total_usage={})
            return
        # stdout closed (the process exited) without turn.completed or
        # turn.failed — e.g. it crashed outright. stderr is the only lead left.
        #
        # The reachable case is a REFUSED RESUME. codex prunes its own rollout
        # files, and `codex exec resume <id>` on an id it no longer has prints
        # "thread/resume failed: no rollout found for thread id ..." as PLAIN
        # TEXT on stderr, exits 0, and writes ZERO JSONL to stdout — not even
        # thread.started, so nothing below ever reset _thread_id. Keeping the
        # dead id would wedge the chat permanently: every later turn resumes the
        # same missing thread and fails identically, with Clear (reset()) the
        # only escape and no hint that it's the fix. Dropping it costs this
        # session's codex-side history — which is already gone — and lets the
        # next turn start a fresh thread on its own.
        self._thread_id = None
        detail = "\n".join(line for line in stderr_lines[-20:] if line).strip()
        yield TextDelta(
            text=f"\n[codex agent error] {detail or 'codex exited without completing the turn'}"
        )
        yield TurnDone(stop_reason="error", total_usage={})
