"""Tests for the loopback MCP tool bridge (imajin.agent.mcp_bridge).

Two layers, per the slice brief:

1. Hermetic (default suite, no socket bound): the ``list_tools``/``call_tool``
   handler methods are called directly. These are the same coroutines the real
   MCP server dispatches to (see ``ImajinMcpBridge.__init__``), so this exercises
   the actual routing/compaction/error-wrapping logic without any network I/O.
2. A real loopback round-trip over ``mcp.client.streamable_http``, using ONE
   cheap real tool (``get_help``) with a FAKE ``tool_caller`` — no napari, no
   codex, no quota, no subprocess. This proves the auth/DNS-rebinding gates
   and the wire format actually work against the real transport, not just our
   model of it.
"""
from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from imajin.agent.mcp_bridge import ImajinMcpBridge
from imajin.tools.registry import iter_tools

_EXPECTED_ENTRIES = [e for e in iter_tools() if e.subagent is None and e.llm]


# -- layer 1: hermetic, no socket -------------------------------------------


def test_list_tools_matches_the_registry_predicate():
    """Same predicate as ClaudeAgentRunner._bridged_entries(): count and names must agree,
    and every advertised tool must carry a real, non-empty JSON Schema (a bare `Tool(...)`
    with an empty inputSchema is what FastMCP's kwargs-introspection would have produced —
    see the class docstring for why this bridge avoids that)."""
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None)

    tools = asyncio.run(bridge._handle_list_tools())

    assert len(tools) == len(_EXPECTED_ENTRIES)
    assert {t.name for t in tools} == {e.name for e in _EXPECTED_ENTRIES}
    assert "get_help" in {t.name for t in tools}
    for t in tools:
        assert isinstance(t.inputSchema, dict) and t.inputSchema, t.name


def test_list_tools_schemas_are_non_empty_and_carry_the_real_description():
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None, tool_names=["get_help"])

    (tool,) = asyncio.run(bridge._handle_list_tools())

    assert tool.name == "get_help"
    real_entry = next(e for e in _EXPECTED_ENTRIES if e.name == "get_help")
    assert tool.description == real_entry.description
    assert isinstance(tool.inputSchema, dict) and tool.inputSchema
    assert tool.inputSchema == real_entry.json_schema


def test_tool_names_narrows_the_listed_set():
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None, tool_names=["get_help"])

    tools = asyncio.run(bridge._handle_list_tools())

    assert [t.name for t in tools] == ["get_help"]


def test_call_tool_routes_to_the_injected_tool_caller():
    calls: list[tuple[str, dict]] = []

    def fake_caller(name: str, **kwargs):
        calls.append((name, kwargs))
        return {"guide_url": "https://example.invalid/guide", "topics": []}

    bridge = ImajinMcpBridge(tool_caller=fake_caller)

    result = asyncio.run(bridge._handle_call_tool("get_help", {"topic": "install_and_run"}))

    assert calls == [("get_help", {"topic": "install_and_run"})]
    assert result.isError is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "example.invalid" in result.content[0].text


def test_call_tool_wraps_a_raised_exception_as_is_error():
    def raising_caller(name: str, **kwargs):
        raise ValueError("no image loaded")

    bridge = ImajinMcpBridge(tool_caller=raising_caller)

    result = asyncio.run(bridge._handle_call_tool("get_help", {}))

    assert result.isError is True
    assert "no image loaded" in result.content[0].text


def test_call_tool_treats_missing_arguments_as_empty_kwargs():
    calls: list[tuple[str, dict]] = []

    def fake_caller(name: str, **kwargs):
        calls.append((name, kwargs))
        return "ok"

    bridge = ImajinMcpBridge(tool_caller=fake_caller)

    result = asyncio.run(bridge._handle_call_tool("get_help", None))

    assert calls == [("get_help", {})]
    assert result.isError is False


def test_port_and_url_raise_before_start():
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None)

    with pytest.raises(RuntimeError):
        _ = bridge.port
    with pytest.raises(RuntimeError):
        _ = bridge.url


def test_token_is_present_and_unique_per_instance():
    a = ImajinMcpBridge(tool_caller=lambda name, **kw: None)
    b = ImajinMcpBridge(tool_caller=lambda name, **kw: None)

    assert isinstance(a.token, str) and len(a.token) > 20
    assert a.token != b.token


def test_stop_before_start_is_a_harmless_no_op():
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None)
    bridge.stop()  # must not raise


# -- layer 2: a real loopback round-trip -------------------------------------


@pytest.fixture
def running_bridge():
    """A live bridge exposing only `get_help`, backed by a FAKE tool_caller.

    No real Imajin tool is ever invoked here (`get_help` names which real tool
    is *advertised*, but the fake tool_caller intercepts every actual call) —
    so this needs no napari session and touches no GPU/Qt state.
    """
    calls: list[tuple[str, dict]] = []

    def fake_caller(name: str, **kwargs):
        calls.append((name, kwargs))
        return {"guide_url": "https://example.invalid/guide", "topics": []}

    bridge = ImajinMcpBridge(tool_caller=fake_caller, tool_names=["get_help"])
    bridge.start()
    try:
        yield bridge, calls
    finally:
        bridge.stop()


def test_loopback_round_trip_with_the_correct_token(running_bridge):
    bridge, calls = running_bridge

    async def _round_trip():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": f"Bearer {bridge.token}"}
        # streamable_http_client (unlike the deprecated streamablehttp_client)
        # takes a pre-configured httpx.AsyncClient rather than raw header kwargs.
        async with httpx.AsyncClient(headers=headers, timeout=10) as http_client:
            async with streamable_http_client(bridge.url, http_client=http_client) as (
                read,
                write,
                _get_sid,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert [t.name for t in listed.tools] == ["get_help"]
                    return await session.call_tool("get_help", {})

    result = asyncio.run(_round_trip())

    assert calls == [("get_help", {})]
    assert result.isError is False
    assert "example.invalid" in result.content[0].text


def test_wrong_token_gets_401(running_bridge):
    bridge, calls = running_bridge

    resp = httpx.get(
        bridge.url, headers={"Authorization": "Bearer not-the-right-token"}, timeout=5
    )

    assert resp.status_code == 401
    assert calls == []  # the fake tool_caller was never reached


def test_missing_token_gets_401(running_bridge):
    bridge, _calls = running_bridge

    resp = httpx.get(bridge.url, timeout=5)

    assert resp.status_code == 401


def test_request_carrying_an_origin_header_is_rejected(running_bridge):
    bridge, calls = running_bridge

    resp = httpx.get(
        bridge.url,
        headers={
            "Authorization": f"Bearer {bridge.token}",
            "Origin": "http://evil.example",
        },
        timeout=5,
    )

    assert resp.status_code == 403
    assert calls == []


def test_request_with_a_foreign_host_header_is_rejected(running_bridge):
    """DNS-rebinding guard, the other half of the Origin check above.

    A page that resolves some attacker-controlled name to 127.0.0.1 reaches this
    port with `Host: evil.example`. TransportSecuritySettings.allowed_hosts is
    pinned to the bridge's own `127.0.0.1:<port>`, so that request must die at
    the transport layer -- before the session manager, and before the tool_caller.
    """
    bridge, calls = running_bridge

    resp = httpx.get(
        bridge.url,
        headers={"Authorization": f"Bearer {bridge.token}", "Host": "evil.example"},
        timeout=5,
    )

    assert resp.status_code == 421
    assert calls == []


def test_the_bridge_is_not_reachable_off_loopback(running_bridge):
    """The listener must be bound to 127.0.0.1, never 0.0.0.0.

    The bearer token is the access control (see the module docstring), but
    defence in depth means the socket should not be dialable from the LAN in the
    first place -- a bind-host regression would otherwise be invisible, since
    every existing test connects over loopback and would keep passing.
    """
    bridge, _calls = running_bridge

    # The UDP-connect trick, not socket.gethostbyname(socket.gethostname()):
    # on this project's WSL dev hosts the latter answers 127.0.1.1 from
    # /etc/hosts, which would silently skip the very assertion this test exists
    # to make. connect() on a UDP socket sends nothing -- it only asks the
    # kernel which local address would route outward.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))  # TEST-NET-1, never routed anywhere
        local_ip = probe.getsockname()[0]
    except OSError:
        local_ip = "127.0.0.1"
    finally:
        probe.close()
    if local_ip.startswith("127."):
        pytest.skip("host has no non-loopback address to dial from")

    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://{local_ip}:{bridge.port}/mcp", timeout=3)


def test_stop_releases_the_port_and_stops_answering():
    """stop() must actually free the listener, not just flip a flag.

    A bridge is started per chat session and stopped on every model switch /
    dock close (ChatDock._release_runner -> CodexAgentRunner.close), so a
    listener that outlived stop() would accumulate one live, token-guarded MCP
    server per switch for the lifetime of the app.
    """
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: None, tool_names=["get_help"])
    bridge.start()
    port = bridge.port
    assert httpx.get(bridge.url, headers={"Authorization": f"Bearer {bridge.token}"}).status_code

    bridge.stop()

    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/mcp", timeout=3)
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        probe.listen(1)
    finally:
        probe.close()


def test_start_after_stop_rebinds_a_working_server():
    """One bridge instance must survive stop() -> start().

    StreamableHTTPSessionManager.run() can only be entered once per instance, so
    a restart that reused the old manager would fail at the second start();
    _build_asgi_app makes a fresh one for exactly this reason, and nothing else
    covers that path.
    """
    bridge = ImajinMcpBridge(tool_caller=lambda name, **kw: "ok", tool_names=["get_help"])
    bridge.start()
    first_port = bridge.port
    bridge.stop()

    bridge.start()
    try:
        assert bridge.port != first_port
        resp = httpx.get(bridge.url, headers={"Authorization": f"Bearer {bridge.token}"}, timeout=5)
        assert resp.status_code != 401
    finally:
        bridge.stop()
