"""In-process MCP tool bridge: exposes Imajin's LLM tool registry over loopback HTTP.

Written for the "Codex (subscription)" chat backend
(:mod:`imajin.agent.providers.codex_agent`). Unlike the Claude Agent SDK, which
gives us an in-process "register this Python callable as a tool" hook
(``claude_agent_sdk.tool()``, see ``claude_agent.py``), the ``codex`` CLI owns its
own agentic loop as an external process and only ever calls tools over a real MCP
connection. This module is that connection's server side: a
``streamable-http`` MCP server, bound to loopback, that turns
``imajin.tools.registry`` entries into MCP ``Tool`` definitions and forwards
``tools/call`` back into this process's live napari session via an injected
``tool_caller``.

SECURITY — read this before touching the bind address or the token check.
Loopback TCP (127.0.0.1) authenticates *machines*, not *processes*: any process
running as this OS user (and, on a shared/misconfigured host, potentially other
local users) can open a socket to ``127.0.0.1:<port>`` and speak MCP to it. There
is no OS-level caller authentication at that layer. The per-instance bearer
token (:attr:`ImajinMcpBridge.token`, a fresh ``secrets.token_urlsafe(32)`` every
time a bridge is built) *is* the access control, enforced by
``_BearerAuthMiddleware`` with a constant-time comparison
(``secrets.compare_digest`` — a plain ``==`` would leak how many leading bytes of
a guess matched via timing). Never widen the bind host past 127.0.0.1, never log
the token, and never skip the comparison "just for a quick local test" — that
test will get copy-pasted into something that isn't loopback-only.
"""
from __future__ import annotations

import asyncio
import contextlib
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, Tool
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

_SERVER_NAME = "imajin"
_START_TIMEOUT = 10.0
_STOP_TIMEOUT = 10.0


async def _placeholder_asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Stand-in ``uvicorn.Config.app`` for the brief window before the real one exists.

    ``TransportSecuritySettings.allowed_hosts`` needs the bound port, which is
    only known *after* ``Config.bind_socket()`` runs (see ``start()``), but
    ``Config`` requires an app at construction. This is swapped for the real app
    before ``config.load()`` ever reads it (that happens inside the daemon
    thread's ``Server.run()``, started after the swap) — reaching this is a bug
    in that ordering, so it fails loud instead of serving a confusing 500.
    """
    raise RuntimeError("ImajinMcpBridge: placeholder ASGI app was never replaced")


def _extract_bearer(header_value: str | None) -> str | None:
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


class _BearerAuthMiddleware:
    """Rejects any HTTP request without the bridge's exact bearer token.

    This is the whole access-control story for this server — see the module
    docstring. Wraps the *entire* Starlette app (there is only one route, so
    there is no legitimate unauthenticated path to carve out) and runs before
    routing, so a missing/wrong token never reaches the MCP session manager.
    Non-HTTP ASGI scopes (``lifespan``) pass straight through: they carry no
    headers and Starlette's own startup/shutdown handshake depends on them.
    """

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        provided = _extract_bearer(Headers(scope=scope).get("authorization"))
        if provided is None or not secrets.compare_digest(provided, self._token):
            await PlainTextResponse("Unauthorized", status_code=401)(scope, receive, send)
            return
        await self._app(scope, receive, send)


class _StreamableHTTPEndpoint:
    """Adapts the session manager to a raw ASGI callable for ``starlette.routing.Route``.

    ``Route.__init__`` special-cases a plain function *or bound method* endpoint
    as ``func(request) -> response`` (``starlette.routing.request_response``),
    which is not the calling convention ``StreamableHTTPSessionManager.handle_request``
    uses (it wants the raw ``(scope, receive, send)`` triple). A callable
    *instance* is neither ``inspect.isfunction`` nor ``inspect.ismethod``, so
    ``Route`` falls through to its "already ASGI" branch instead — the same
    trick FastMCP's own ``StreamableHTTPASGIApp`` uses for exactly this reason.
    """

    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


class ImajinMcpBridge:
    """In-process streamable-HTTP MCP server exposing Imajin's tools on loopback.

    Built on mcp.server.lowlevel.Server (NOT FastMCP: FastMCP.tool()/add_tool()
    derive inputSchema by introspecting the callable and accept no explicit JSON
    schema, so Imajin's existing ToolEntry.json_schema would be thrown away and a
    generic **kwargs handler would advertise an empty schema).
    """

    def __init__(
        self,
        tool_caller: Callable[..., Any],
        *,
        tool_names: list[str] | None = None,
    ) -> None:
        self._tool_caller = tool_caller
        self._tool_names = tool_names
        self._token = secrets.token_urlsafe(32)
        self._server = Server(_SERVER_NAME)
        self._server.list_tools()(self._handle_list_tools)
        self._server.call_tool()(self._handle_call_tool)
        self._session_manager: StreamableHTTPSessionManager | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # -- MCP protocol handlers ------------------------------------------------
    # Registered as bound methods in __init__ above (not via `@server.list_tools()`
    # decorator syntax) so tests can call them directly — no socket, no event
    # loop wiring beyond a bare `asyncio.run(...)` — and still get the exact
    # handlers the real server dispatches to.

    def _entries(self) -> list[Any]:
        from imajin.tools.registry import iter_tools  # deferred: see _handle_call_tool

        # Same selection as ClaudeAgentRunner._bridged_entries() / tools_for_anthropic():
        # top-level LLM tools, no subagent-only tools. Keeping this predicate
        # identical across chat backends means the model sees the same tool
        # surface no matter which backend the user picked.
        entries = [e for e in iter_tools() if e.subagent is None and e.llm]
        if self._tool_names is not None:
            wanted = set(self._tool_names)
            entries = [e for e in entries if e.name in wanted]
        return entries

    async def _handle_list_tools(self) -> list[Tool]:
        return [
            Tool(name=e.name, description=e.description, inputSchema=e.json_schema)
            for e in self._entries()
        ]

    async def _handle_call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        # Deferred so `import imajin.agent.mcp_bridge` alone never pulls in the
        # whole tool package (numpy/torch-backed, ~30 modules) or the runner —
        # only a bridge that actually gets used to call a tool pays for them.
        from imajin.agent.runner import _compact_tool_result

        loop = asyncio.get_running_loop()
        try:
            # MANDATORY offload, not an optimization: this coroutine runs on the
            # bridge's one and only event loop. A synchronous call here would
            # block that loop — and therefore every other in-flight MCP request,
            # including codex's next tools/list — for as long as the tool takes,
            # and Imajin tools marshal onto the Qt main thread and can run for
            # minutes (cellpose segmentation, batch jobs). Mirrors
            # ClaudeAgentRunner._make_handler in claude_agent.py.
            result = await loop.run_in_executor(
                None, lambda: self._tool_caller(name, **(arguments or {}))
            )
            text = _compact_tool_result(name, result)
            return CallToolResult(content=[TextContent(type="text", text=text)])
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            return CallToolResult(
                content=[TextContent(type="text", text=f"ERROR: {exc}")],
                isError=True,
            )

    # -- ASGI app construction --------------------------------------------------

    def _build_asgi_app(self, port: int) -> Any:
        # A fresh StreamableHTTPSessionManager every call: the class's own
        # docstring is explicit that one instance's .run() context can only be
        # entered once ("create a new instance if you need to run again"), and
        # start()/stop() must support being called more than once across a
        # bridge's lifetime (one bridge per chat session, possibly reset).
        self._session_manager = StreamableHTTPSessionManager(
            app=self._server,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[f"127.0.0.1:{port}"],
                # No legitimate MCP client sends an Origin header (that's a
                # browser concept); rejecting any request that carries one is
                # the correct policy against a web page DNS-rebinding to this
                # port from inside the user's browser.
                allowed_origins=[],
            ),
        )
        manager = self._session_manager
        route = Route("/mcp", _StreamableHTTPEndpoint(manager), methods=["GET", "POST", "DELETE"])

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette):
            async with manager.run():
                yield

        starlette_app = Starlette(routes=[route], lifespan=lifespan)
        return _BearerAuthMiddleware(starlette_app, self._token)

    # -- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        """Bind 127.0.0.1:0 and start serving on a daemon thread. Idempotent."""
        if self._thread is not None:
            return
        config = uvicorn.Config(
            app=_placeholder_asgi_app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        # Bind now, while `config.app` is still the placeholder: the security
        # settings below need the *port*, which only exists once the OS has
        # assigned one, and finding that out via a throwaway probe-bind-close
        # socket would leave a window for another process to grab the same
        # port before our real bind. `bind_socket()` IS the real listening
        # socket — we hand this exact object to `Server.run()` below, so
        # there is no second bind and no race.
        sock = config.bind_socket()
        port = sock.getsockname()[1]
        config.app = self._build_asgi_app(port)

        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="imajin-mcp-bridge",
            daemon=True,  # never blocks process exit if shutdown hangs (see stop())
        )
        self._uvicorn_server = server
        self._port = port
        self._thread = thread
        thread.start()
        try:
            self._wait_until_started(_START_TIMEOUT)
        except Exception:
            self.stop()
            raise

    def _wait_until_started(self, timeout: float) -> None:
        """Block until uvicorn is actually accepting connections, or raise.

        `bind_socket()` only reserves the port; the socket starts *listening*
        inside the daemon thread's own event loop (`Server.startup()`), which
        runs asynchronously after the thread starts. Returning from `start()`
        before that completes would hand a caller a URL nothing answers yet —
        indistinguishable from a hang until it times out downstream instead of
        here, where the real cause is still known.
        """
        server, thread = self._uvicorn_server, self._thread
        assert server is not None and thread is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server.started:
                return
            if not thread.is_alive():
                raise RuntimeError("Imajin MCP bridge thread exited before starting up")
            time.sleep(0.01)
        raise RuntimeError(f"Imajin MCP bridge did not start within {timeout}s")

    def stop(self) -> None:
        """Shut the listener down and join its thread. Idempotent.

        Callers should make sure any connected client (the `codex` subprocess)
        is already gone before calling this: uvicorn's graceful shutdown waits
        for open connections to finish, `timeout_graceful_shutdown` defaults to
        None (wait forever), and a still-attached streamable-http client can
        hold a connection open indefinitely. The listener thread is a daemon
        thread regardless, so a stuck shutdown cannot block process exit — only
        this call, bounded here by `_STOP_TIMEOUT`.
        """
        server, thread = self._uvicorn_server, self._thread
        self._uvicorn_server = None
        self._thread = None
        self._session_manager = None
        self._port = None
        if server is None:
            return
        server.should_exit = True
        if thread is not None:
            thread.join(timeout=_STOP_TIMEOUT)

    # -- accessors ----------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def token(self) -> str:
        return self._token

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("ImajinMcpBridge.start() has not been called")
        return self._port
