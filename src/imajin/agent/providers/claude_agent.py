"""Subscription-backed agent that runs on the local Claude Code login.

Unlike :class:`~imajin.agent.providers.anthropic.AnthropicProvider` and
:class:`~imajin.agent.providers.openai_compat.OpenAICompatProvider`, this is **not**
a :class:`~imajin.agent.providers.base.Provider`. Those providers implement a
single-model-turn ``stream()`` and let :class:`~imajin.agent.runner.AgentRunner`
own the tool loop. The Claude Agent SDK owns its *own* agentic loop (it drives the
`claude` CLI, which executes tools itself), so it cannot sit behind ``stream()``.

Instead this class fuses provider + runner: it presents the same surface the chat
dock already drives on ``AgentRunner`` — ``turn()`` yielding the shared ``RunEvent``
types, plus ``reset()`` / ``cancel()`` — while internally delegating the loop to the
SDK. Imajin's own tools are bridged in as an in-process MCP server so the agent's
loop calls back into this process; the SDK's message stream is translated back into
Imajin ``RunEvent`` objects for the UI.

Auth: the `claude` CLI resolves its own credentials, so **no API key is needed** —
it uses whatever the user logged into Claude Code with (a Pro/Max subscription).
This is the Anthropic-sanctioned way to use a subscription programmatically: the
Agent SDK is a documented "surface that wraps the CLI". We never read or forward
the OAuth token ourselves (doing so against the raw Messages API would violate the
Consumer Terms) — the CLI holds it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from imajin.agent.providers.base import TextDelta, ToolUse, ToolUseStart
from imajin.agent.runner import ToolResult, TurnDone

# Bridged Imajin tools live under this MCP server name; the model sees each tool as
# ``mcp__<server>__<tool>``.
_MCP_SERVER = "imajin"
_TOOL_PREFIX = f"mcp__{_MCP_SERVER}__"

# API-key env vars outrank subscription OAuth in the CLI's auth precedence, so a
# stray key in the environment would silently bill the API instead of the
# subscription. We strip them for the duration of a turn (see _force_subscription_env).
_AUTH_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_ENV_LOCK = threading.Lock()

# Built-in Claude Code tools we never want an embedded, napari-bound agent to reach.
# permission_mode="dontAsk" already denies anything not in allowed_tools; this just
# stops the model from wasting turns attempting them.
_DISALLOWED_BUILTINS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
]


def _sdk():
    """Import the Claude Agent SDK lazily so the dependency stays optional."""
    import claude_agent_sdk  # noqa: PLC0415

    return claude_agent_sdk


def subscription_available() -> tuple[bool, str | None]:
    """Whether the subscription path is usable, and a short reason if not.

    Checked without importing the SDK or spawning the CLI so it is cheap enough
    for the model-picker status probe.
    """
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return False, "SDK missing"
    if shutil.which("claude") is None:
        return False, "claude not found"
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True, None
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    creds = (Path(config_dir) if config_dir else Path.home() / ".claude") / ".credentials.json"
    if creds.exists():
        return True, None
    return False, "not logged in"


def _strip_ns(name: str) -> str:
    return name[len(_TOOL_PREFIX):] if name.startswith(_TOOL_PREFIX) else name


def _flatten_tool_result(content: Any) -> str:
    """Reduce an SDK ToolResultBlock ``content`` (str | list[block]) to display text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _map_usage(raw: Any) -> dict[str, int]:
    usage: dict[str, int] = {}
    if isinstance(raw, dict):
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = raw.get(key)
            if isinstance(value, int):
                usage[key] = value
    return usage


def _translate_message(message: Any, id_to_name: dict[str, str]) -> list[Any]:
    """Translate one SDK message into Imajin ``RunEvent`` objects.

    Dispatch is by attribute rather than ``isinstance`` so it stays testable
    without importing the SDK: content blocks in this SDK version carry no ``type``
    field, but their fields are distinctive (a tool_result has ``tool_use_id``, a
    tool_use has ``id``/``name``/``input``, text has ``text``). Assistant and user
    messages both expose ``content``; a result message exposes ``num_turns``.
    """
    events: list[Any] = []

    # ResultMessage — end of the turn.
    if hasattr(message, "num_turns") and hasattr(message, "session_id"):
        subtype = getattr(message, "subtype", None)
        stop = getattr(message, "stop_reason", None) or (
            "end_turn" if subtype == "success" else (subtype or "end_turn")
        )
        events.append(
            TurnDone(stop_reason=str(stop), total_usage=_map_usage(getattr(message, "usage", None)))
        )
        return events

    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return events

    for block in content:
        if hasattr(block, "tool_use_id"):  # ToolResultBlock (in a UserMessage)
            tool_use_id = block.tool_use_id
            events.append(
                ToolResult(
                    tool_use_id=tool_use_id,
                    name=id_to_name.get(tool_use_id, "tool"),
                    output=_flatten_tool_result(getattr(block, "content", "")),
                    is_error=bool(getattr(block, "is_error", False)),
                )
            )
        elif hasattr(block, "input") and hasattr(block, "name") and hasattr(block, "id"):
            # ToolUseBlock (in an AssistantMessage)
            name = _strip_ns(block.name)
            id_to_name[block.id] = name
            events.append(ToolUseStart(id=block.id, name=name))
            events.append(ToolUse(id=block.id, name=name, input=dict(block.input or {})))
        elif hasattr(block, "text"):  # TextBlock
            if block.text:
                events.append(TextDelta(text=block.text))
        # ThinkingBlock and anything else: ignored for the UI stream.

    return events


@contextmanager
def _force_subscription_env():
    """Temporarily remove API-key env vars so the CLI falls back to subscription OAuth.

    The SDK spawns the CLI with ``{**os.environ, **options.env}``, so a value that is
    present in the parent environment cannot be removed via ``options.env`` (only
    overridden, and an empty ``ANTHROPIC_API_KEY`` would break the *working* no-key
    case). We therefore pop the keys from the process environment around the turn.
    Serialized by a lock and a no-op when the keys are absent (the common case).
    """
    with _ENV_LOCK:
        saved = {k: os.environ.pop(k) for k in _AUTH_ENV_KEYS if k in os.environ}
        try:
            yield
        finally:
            os.environ.update(saved)


# Marks the end of a turn's event stream across the loop→worker queue.
_SENTINEL = object()


def _explain_connect_failure(exc: Exception) -> Exception:
    """Turn the SDK's misleading 'Claude Code not found' into the real reason.

    The SDK catches ``FileNotFoundError`` from the process spawn and blames the
    CLI path. On Windows that same exception is what a command line over 32,767
    characters produces (WinError 206), so a perfectly present 230 MB claude.exe
    gets reported as missing. If the path it names is actually there, say what is
    really wrong.
    """
    message = str(exc)
    marker = "Claude Code not found at: "
    if marker not in message:
        return exc
    cli_path = message.split(marker, 1)[1].strip()
    if not cli_path or not Path(cli_path).exists():
        return exc  # genuinely missing — the SDK's message is correct
    return RuntimeError(
        f"the Claude Code CLI at {cli_path} exists but could not be launched. "
        "On Windows this is almost always a command line over the 32,767-character "
        "limit — the system prompt plus the bridged tool list. Shorten the system "
        "prompt or reduce the number of bridged tools."
    )


class ClaudeAgentRunner:
    """Runner backed by the local Claude Code subscription via the Claude Agent SDK.

    Owns **one** persistent asyncio loop on a dedicated daemon thread and **one**
    persistent :class:`ClaudeSDKClient` for its whole lifetime. Each ``turn()``
    reuses that connection (``query`` + ``receive_response``) and streams events
    back to the chat worker thread via a queue. This replaces the old
    new-loop-per-turn + one-shot ``query()`` design, which — because it closed a
    loop while the SDK's subprocess/tasks were still pending and reused a
    loop-bound MCP server across loops — orphaned ``claude`` subprocesses and
    eventually froze the chat (worst on Windows' ProactorEventLoop).
    """

    name = "claude-agent"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tool_caller: Any | None = None,
        max_turns: int = 24,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._tool_caller = tool_caller
        self._cancelled = False
        self._session_id: str | None = None
        self._server: Any | None = None
        self._allowed: list[str] = []
        self._cwd = tempfile.gettempdir()
        self._prompt_file: Path | None = None
        # Persistent async machinery (created lazily on the first turn).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any | None = None

    # -- lifecycle (mirrors AgentRunner) ------------------------------------

    def cancel(self) -> None:
        self._cancelled = True
        client, loop = self._client, self._loop
        if client is not None and loop is not None:
            # Best-effort: ask the running turn to stop. Fire-and-forget.
            try:
                asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
            except Exception:  # noqa: BLE001
                pass

    def reset(self) -> None:
        # Drop the live connection so the next turn starts a fresh conversation.
        self._cancelled = False
        self._session_id = None
        self._disconnect()

    def close(self) -> None:
        """Tear down the connection, loop, and thread. Call when discarding the runner."""
        self._disconnect()
        loop, thread = self._loop, self._thread
        self._loop, self._thread = None, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        if loop is not None:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    # -- tool bridge ---------------------------------------------------------

    def _bridged_entries(self) -> list[Any]:
        from imajin.tools.registry import iter_tools

        # Same selection as tools_for_anthropic(): top-level LLM tools, no subagents.
        return [e for e in iter_tools() if e.subagent is None and e.llm]

    def _make_handler(self, tool_name: str):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            from imajin.agent.runner import _compact_tool_result
            from imajin.tools import call_tool

            caller = self._tool_caller or call_tool
            loop = asyncio.get_running_loop()
            try:
                # Run the (blocking, main-thread-marshalling) tool call off the event
                # loop so the SDK transport keeps draining while the tool executes.
                result = await loop.run_in_executor(None, lambda: caller(tool_name, **args))
                text = _compact_tool_result(tool_name, result)
                return {"content": [{"type": "text", "text": text}]}
            except Exception as exc:  # noqa: BLE001 - surface as a tool error to the model
                return {
                    "content": [{"type": "text", "text": f"ERROR: {exc}"}],
                    "is_error": True,
                }

        return handler

    def _build_server(self) -> tuple[Any, list[str]]:
        if self._server is not None:
            return self._server, self._allowed
        sdk = _sdk()
        sdk_tools = []
        allowed: list[str] = []
        for entry in self._bridged_entries():
            sdk_tool = sdk.tool(entry.name, entry.description, entry.json_schema)(
                self._make_handler(entry.name)
            )
            sdk_tools.append(sdk_tool)
            allowed.append(f"{_TOOL_PREFIX}{entry.name}")
        self._server = sdk.create_sdk_mcp_server(name=_MCP_SERVER, version="0.1.0", tools=sdk_tools)
        self._allowed = allowed
        return self._server, self._allowed

    def _write_prompt_file(self) -> Path:
        """Spill the system prompt to a file so it stays off the command line.

        The SDK passes ``--system-prompt <text>`` as one argv element. Windows
        caps a whole command line at 32,767 characters, and the system prompt
        alone is ~30k before the ~3k of ``--allowedTools`` for Imajin's 130
        bridged tools — so inlining it overflows, CreateProcess fails with
        WinError 206, and Python raises FileNotFoundError, which the SDK reports
        as the thoroughly misleading "Claude Code not found at: <path>".

        ``--system-prompt-file`` takes the text out of the argv budget entirely,
        so the limit stops depending on how long the prompt has grown.
        """
        if self._prompt_file is not None and self._prompt_file.exists():
            return self._prompt_file
        fd, name = tempfile.mkstemp(prefix="imajin_system_prompt_", suffix=".md")
        os.close(fd)
        path = Path(name)
        path.write_text(self.system_prompt, encoding="utf-8")
        self._prompt_file = path
        return path

    def _cleanup_prompt_file(self) -> None:
        path, self._prompt_file = self._prompt_file, None
        if path is None:
            return
        try:
            path.unlink()
        except OSError:  # noqa: PERF203 - best-effort cleanup
            pass

    def _build_options(self, server: Any, allowed: list[str]) -> Any:
        sdk = _sdk()
        return sdk.ClaudeAgentOptions(
            model=self.model,
            system_prompt={"type": "file", "path": str(self._write_prompt_file())},
            mcp_servers={_MCP_SERVER: server},
            allowed_tools=allowed,
            disallowed_tools=list(_DISALLOWED_BUILTINS),
            permission_mode="dontAsk",  # deny anything not pre-approved, without prompting
            setting_sources=[],  # ignore ambient ~/.claude and project settings
            max_turns=self.max_turns,
            cwd=self._cwd,
            resume=self._session_id,
        )

    # -- persistent loop + connection ---------------------------------------

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="claude-agent-loop", daemon=True
        )
        thread.start()
        self._loop, self._thread = loop, thread

    def _call(self, coro: Any, timeout: float | None = None) -> Any:
        """Run ``coro`` on the persistent loop from this (worker) thread; block."""
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _ensure_connected(self) -> None:
        self._ensure_loop()
        if self._client is not None:
            return
        sdk = _sdk()
        server, allowed = self._build_server()
        options = self._build_options(server, allowed)
        client = sdk.ClaudeSDKClient(options=options)
        # The CLI subprocess is spawned during connect(); strip API-key env vars
        # for that window so it authenticates against the subscription OAuth.
        with _force_subscription_env():
            try:
                self._call(client.connect())
            except Exception as exc:  # noqa: BLE001
                raise _explain_connect_failure(exc) from exc
        self._client = client

    def _disconnect(self) -> None:
        client = self._client
        self._client = None
        self._cleanup_prompt_file()
        if client is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                client.disconnect(), self._loop
            ).result(timeout=10)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    # -- turn driving --------------------------------------------------------

    async def _adrive_turn(self, user_text: str, q: queue.Queue) -> None:
        id_to_name: dict[str, str] = {}
        try:
            await self._client.query(user_text)
            async for message in self._client.receive_response():
                if self._cancelled:
                    break
                if hasattr(message, "num_turns") and hasattr(message, "session_id"):
                    self._session_id = getattr(message, "session_id", None) or self._session_id
                for event in _translate_message(message, id_to_name):
                    q.put(event)
        except Exception as exc:  # noqa: BLE001 - report in-stream, don't crash the worker
            q.put(TextDelta(text=f"\n[subscription agent error] {type(exc).__name__}: {exc}"))
            q.put(TurnDone(stop_reason="error", total_usage={}))
            # A broken connection is the usual culprit — drop it so the next turn
            # reconnects fresh (also clears any stale resumed session id).
            client, self._client, self._session_id = self._client, None, None
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            q.put(_SENTINEL)

    def turn(self, user_text: str) -> Iterator[Any]:
        """Drive one turn, yielding ``RunEvent`` objects synchronously.

        Connect lazily (once), run the turn on the persistent loop, and stream its
        events back through a queue to this (chat worker) thread. The connection —
        and its ``claude`` subprocess — survives across turns and is torn down only
        by :meth:`reset` / :meth:`close`.
        """
        self._cancelled = False
        try:
            self._ensure_connected()
        except Exception as exc:  # noqa: BLE001 - connect failed; report and bail
            self._disconnect()
            yield TextDelta(text=f"\n[subscription agent error] {type(exc).__name__}: {exc}")
            yield TurnDone(stop_reason="error", total_usage={})
            return

        q: queue.Queue = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._adrive_turn(user_text, q), self._loop
        )
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            try:
                future.result(timeout=5)
            except Exception:  # noqa: BLE001 - already surfaced in-stream
                pass
