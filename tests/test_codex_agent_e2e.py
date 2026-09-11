"""End-to-end gate for the Codex (subscription) backend: a REAL `codex exec` turn.

Everything else that covers this feature is a model of it. tests/test_codex_agent_runner.py
replays captured JSONL through a fake Popen; tests/test_mcp_bridge.py drives the bridge
with the Python MCP client. Both would keep passing if `codex` stopped honouring
`-c mcp_servers.imajin.bearer_token_env_var`, if its event schema drifted, or if the
approval flag stopped auto-approving MCP tool calls -- that last one is the dangerous
case, because codex reports it as `turn.completed` with exit code 0 and an empty stderr
(see NOAPPROVAL_JSONL in test_codex_agent_runner.py). Only a real turn catches those.

THE ASSERTION THAT MATTERS is `invocations` -- that the Python handler in *this*
process was actually entered against the *live* napari viewer. Asserting only on the
text codex reports back would have passed the probe run where the tool was silently
auto-cancelled and codex cheerfully echoed the cancellation message as its answer.

Gated twice on purpose. `@pytest.mark.integration` is the house marker for
"needs an external service", and the skipif means the default suite never spawns
codex even though nothing in this repo's pytest config deselects that marker (there
is no `addopts = -m 'not integration'`; tests/test_anthropic_integration.py uses the
same marker+skipif pairing). Running this costs the operator real ChatGPT
subscription quota -- roughly 50k input / 100 output tokens per run.

    IMAJIN_CODEX_E2E=1 IMAJIN_RESULTS_DIR=$(mktemp -d) \
        .venv/bin/python -m pytest tests/test_codex_agent_e2e.py -q -p no:cacheprovider
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("IMAJIN_CODEX_E2E"),
        reason="spends real codex subscription quota; set IMAJIN_CODEX_E2E=1 to run",
    ),
]

# Deliberately unguessable and absent from every prompt this repo ships, so
# "the answer names the layer" cannot be satisfied by the model repeating
# something it saw in the system prompt instead of calling the tool.
LAYER_NAME = "imajin_e2e_layer_zq7"


@pytest.fixture
def live_viewer(qapp):
    """A REAL napari viewer holding one layer, registered as Imajin's session.

    Not the `viewer` fixture from conftest: that one hands back a ``_FakeViewer``
    whenever QT_QPA_PLATFORM is "offscreen", which is exactly how this suite runs.
    The whole point of this file is that codex reaches a real napari session, so
    it builds its own. Torn down by clearing layers rather than close(), for the
    reason conftest already documents (napari/Qt 6 can abort in QOpenGLWidget
    teardown on offscreen backends).
    """
    import napari
    import numpy as np

    from imajin import session as state

    v = napari.Viewer(show=False)
    v.add_image(np.zeros((3, 8, 8), dtype="uint16"), name=LAYER_NAME, scale=(1.0, 0.25, 0.25))
    state.set_viewer(v)
    yield v
    state.set_viewer(None)
    v.layers.clear()


def test_a_real_codex_turn_calls_a_real_imajin_tool_against_the_live_viewer(live_viewer):
    from imajin.agent.providers.base import TextDelta
    from imajin.agent.providers.codex_agent import CodexAgentRunner, codex_available
    from imajin.agent.runner import ToolResult, TurnDone
    from imajin.tools import call_tool

    usable, reason = codex_available()
    if not usable:
        pytest.skip(f"codex not usable here: {reason}")

    main_thread = threading.get_ident()
    invocations: list[tuple[str, dict, int]] = []

    def counting_caller(name: str, **kwargs):
        invocations.append((name, dict(kwargs), threading.get_ident()))
        return call_tool(name, **kwargs)

    runner = CodexAgentRunner(
        model="gpt-5.6-sol",
        system_prompt=(
            "You are Imajin's analysis assistant, wired to a LIVE napari session. "
            "One tool is available from the `imajin` MCP server: list_layers. "
            "When asked about layers you MUST call that tool and report the real "
            "layer name(s) it returns, verbatim. Never guess. One sentence."
        ),
        tool_caller=counting_caller,
    )
    # Scope the bridge to ONE cheap read-only tool. The production runner
    # advertises all ~107; narrowing keeps this turn's context (and so the
    # operator's quota) small without changing any of the plumbing under test.
    runner._tool_names = lambda: ["list_layers"]

    argvs: list[list[str]] = []
    real_popen_init = subprocess.Popen.__init__

    def spy(self, args, *a, **kw):
        argvs.append([str(x) for x in args])
        return real_popen_init(self, args, *a, **kw)

    subprocess.Popen.__init__ = spy
    started = time.monotonic()
    try:
        events = list(
            runner.turn("Which layers are currently loaded in the viewer? Name them exactly.")
        )
        elapsed = time.monotonic() - started
        proc_after_turn = runner._proc
        token = runner._bridge.token
        port = runner._bridge.port
    finally:
        subprocess.Popen.__init__ = real_popen_init
        runner.close()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    detail = f"\nelapsed={elapsed:.1f}s\nevents={events}\ntext={text!r}"

    # 1. The tool really ran HERE, in this process, against the live viewer.
    #    This is the assertion the silent auto-cancel failure mode breaks; every
    #    other one below can be satisfied by codex talking about the tool.
    assert [(n, k) for n, k, _ in invocations] == [("list_layers", {})], detail
    assert invocations[0][2] != main_thread, (
        "the bridge must offload tool calls off its event loop thread" + detail
    )

    # 2. It surfaced as a completed mcp_tool_call, translated into a ToolResult.
    results = [e for e in events if isinstance(e, ToolResult)]
    assert [r.name for r in results] == ["list_layers"], detail
    assert results[0].is_error is False, detail
    assert LAYER_NAME in results[0].output, detail

    # 3. The model's own answer names the real layer, so the tool output
    #    actually made it back into codex's context and not just into our log.
    assert LAYER_NAME in text, detail

    # 4. The turn terminated cleanly rather than hanging or being killed.
    dones = [e for e in events if isinstance(e, TurnDone)]
    assert [d.stop_reason for d in dones] == ["end_turn"], detail
    assert dones[0].total_usage.get("input_tokens", 0) > 0, detail
    assert proc_after_turn is None, "turn() must clear _proc once the process is reaped" + detail

    # 5. The bearer token never entered any child's argv -- on Linux that is
    #    /proc/<pid>/cmdline, world-readable to every local user. It travels in
    #    the child environment instead (see codex_agent._TOKEN_ENV_VAR).
    assert argvs, "no subprocess was spawned at all" + detail
    assert not any(token in arg for argv in argvs for arg in argv), "TOKEN LEAKED INTO ARGV"
    exec_argv = next(a for a in argvs if "exec" in a)
    assert exec_argv[0] == shutil.which("codex"), "must spawn the resolved absolute path"
    assert 'mcp_servers.imajin.default_tools_approval_mode="approve"' in exec_argv, (
        "without this flag every MCP tool call is silently auto-cancelled" + detail
    )

    # 6. close() released the loopback listener rather than leaking one per session.
    import httpx

    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/mcp", timeout=3)
    assert not [t for t in threading.enumerate() if t.name == "imajin-mcp-bridge"], detail
