from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_registry():
    from imajin.tools import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_job_dock_lists_jobs(qtbot) -> None:
    from qtpy.QtCore import Qt

    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import tool
    from imajin.ui.job_dock import JobDock

    @tool()
    def add(a: int, b: int = 1) -> int:
        return a + b

    service = ToolExecutionService()
    dock = JobDock(execution_service=service)
    qtbot.addWidget(dock)

    job = service.submit_tool("add", {"a": 2}, source="manual", wait=True)
    dock.refresh()

    assert dock.table.rowCount() == 1
    assert dock.table.item(0, 0).text() == "complete"
    assert dock.table.item(0, 2).text() == "add"
    assert dock.table.item(0, 0).data(Qt.ItemDataRole.UserRole) == job.job_id


def test_manual_callable_submits_job() -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import get_tool, tool
    from imajin.ui.manual_dock import _manual_callable

    @tool()
    def add(a: int, b: int = 1) -> int:
        return a + b

    service = ToolExecutionService()
    wrapped = _manual_callable(get_tool("add"), service)

    result = wrapped(a=2, b=3)

    assert result["tool"] == "add"
    assert result["status"] == "complete"
    [job] = service.list_jobs()
    assert job.result == 5
