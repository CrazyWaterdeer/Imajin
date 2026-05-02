from __future__ import annotations

import threading

import pytest


def test_call_on_same_thread_runs_directly(qapp, monkeypatch) -> None:
    from imajin.agent import qt_tool_runner

    captured: list[tuple[str, dict]] = []

    def fake_call_tool(name, **kwargs):
        captured.append((name, kwargs))
        return {"ok": True, "n": name}

    # Patch on the module qt_tool_runner imports from.
    import imajin.tools as imajin_tools

    monkeypatch.setattr(imajin_tools, "call_tool", fake_call_tool, raising=True)

    runner = qt_tool_runner.MainThreadToolRunner()
    out = runner.call("list_layers", x=1)
    assert out == {"ok": True, "n": "list_layers"}
    assert captured == [("list_layers", {"x": 1})]


def test_call_from_worker_thread_marshals_to_main(qapp, monkeypatch, qtbot) -> None:
    """Calling from a non-main thread must execute the tool on the runner's
    thread (main here) — verified by capturing the executing thread ident."""
    from imajin.agent import qt_tool_runner
    import imajin.tools as imajin_tools

    main_ident = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_call_tool(name, **kwargs):
        seen["thread"] = threading.get_ident()
        return name

    monkeypatch.setattr(imajin_tools, "call_tool", fake_call_tool, raising=True)

    runner = qt_tool_runner.MainThreadToolRunner()

    result_holder: dict[str, object] = {}
    err_holder: dict[str, object] = {}

    def worker():
        try:
            result_holder["v"] = runner.call("main_only", do_3D=True)
        except Exception as e:
            err_holder["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Pump the Qt event loop until the worker finishes (handles the queued
    # cross-thread signal). qtbot.waitUntil keeps processEvents going.
    qtbot.waitUntil(lambda: not t.is_alive(), timeout=2000)

    assert err_holder == {}
    assert result_holder["v"] == "main_only"
    assert seen["thread"] == main_ident, "tool ran in wrong thread"


def test_worker_tool_runs_on_worker_thread(qapp, monkeypatch, qtbot) -> None:
    from imajin.agent import qt_tool_runner
    import imajin.tools as imajin_tools

    main_ident = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_call_tool(name, **kwargs):
        seen["thread"] = threading.get_ident()
        return name

    monkeypatch.setattr(imajin_tools, "call_tool", fake_call_tool, raising=True)
    runner = qt_tool_runner.MainThreadToolRunner()

    result_holder: dict[str, object] = {}

    def worker():
        result_holder["v"] = runner.call("cellpose_sam", do_3D=True)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    qtbot.waitUntil(lambda: not t.is_alive(), timeout=2000)

    assert result_holder["v"] == "cellpose_sam"
    assert seen["thread"] != main_ident


def test_tool_exception_propagates_to_caller(qapp, monkeypatch, qtbot) -> None:
    from imajin.agent import qt_tool_runner
    import imajin.tools as imajin_tools

    def fake_call_tool(name, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(imajin_tools, "call_tool", fake_call_tool, raising=True)

    runner = qt_tool_runner.MainThreadToolRunner()

    err_holder: dict[str, object] = {}

    def worker():
        try:
            runner.call("anything")
        except Exception as e:
            err_holder["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    qtbot.waitUntil(lambda: not t.is_alive(), timeout=2000)

    assert isinstance(err_holder.get("e"), ValueError)
    assert "boom" in str(err_holder["e"])


def test_runner_uses_injected_tool_caller() -> None:
    from imajin.agent.runner import AgentRunner

    captured: list = []

    def my_caller(name, **kwargs):
        captured.append((name, kwargs))
        return {"ran": name}

    # Smoke construct — we don't actually drive a turn here; just verify the
    # tool_caller is stored and overrides the default.
    runner = AgentRunner(provider=None, system_prompt="x", tool_caller=my_caller)
    assert runner._tool_caller is my_caller
