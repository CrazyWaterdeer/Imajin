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


class ClaudeAgentRunner:
    """Runner backed by the local Claude Code subscription via the Claude Agent SDK."""

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

    # -- lifecycle (mirrors AgentRunner) ------------------------------------

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        # Drop the resumed CLI session so the next turn starts a fresh conversation.
        self._session_id = None
        self._cancelled = False

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

    def _build_options(self, server: Any, allowed: list[str]) -> Any:
        sdk = _sdk()
        return sdk.ClaudeAgentOptions(
            model=self.model,
            system_prompt=self.system_prompt,
            mcp_servers={_MCP_SERVER: server},
            allowed_tools=allowed,
            disallowed_tools=list(_DISALLOWED_BUILTINS),
            permission_mode="dontAsk",  # deny anything not pre-approved, without prompting
            setting_sources=[],  # ignore ambient ~/.claude and project settings
            max_turns=self.max_turns,
            cwd=self._cwd,
            resume=self._session_id,
        )

    # -- turn driving --------------------------------------------------------

    async def _arun(self, user_text: str) -> Any:
        server, allowed = self._build_server()
        options = self._build_options(server, allowed)
        id_to_name: dict[str, str] = {}
        try:
            async for message in _sdk().query(prompt=user_text, options=options):
                if self._cancelled:
                    break
                if hasattr(message, "num_turns") and hasattr(message, "session_id"):
                    self._session_id = getattr(message, "session_id", None) or self._session_id
                for event in _translate_message(message, id_to_name):
                    yield event
        except Exception as exc:  # noqa: BLE001 - report in-stream, don't crash the worker
            self._session_id = None  # a stale resumed session is the usual culprit
            yield TextDelta(text=f"\n[subscription agent error] {type(exc).__name__}: {exc}")
            yield TurnDone(stop_reason="error", total_usage={})

    def turn(self, user_text: str) -> Iterator[Any]:
        """Drive one turn, yielding ``RunEvent`` objects synchronously.

        The SDK is async and the chat dock consumes a sync generator on a worker
        thread, so we step the async generator with a per-turn event loop. A fresh
        loop per turn keeps this correct even if successive turns run on different
        worker threads.
        """
        self._cancelled = False
        loop = asyncio.new_event_loop()
        agen = self._arun(user_text)
        try:
            asyncio.set_event_loop(loop)
            with _force_subscription_env():
                while True:
                    try:
                        event = loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
                    yield event
        finally:
            try:
                loop.run_until_complete(agen.aclose())
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            asyncio.set_event_loop(None)
            loop.close()
