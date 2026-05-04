from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def isolated_registry():
    from imajin.tools import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_submit_tool_captures_success() -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import tool

    @tool()
    def add(a: int, b: int = 1) -> int:
        return a + b

    service = ToolExecutionService()
    job = service.submit_tool("add", {"a": 2, "b": 3}, source="manual", wait=True)

    assert job.status == "complete"
    assert job.result == 5
    assert job.error is None
    assert service.list_jobs() == [job]


def test_submit_tool_captures_failure() -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import tool

    @tool()
    def explode() -> None:
        raise ValueError("boom")

    service = ToolExecutionService()
    job = service.submit_tool("explode", {}, source="manual", wait=True)

    assert job.status == "failed"
    assert "ValueError: boom" in (job.error or "")


def test_call_tool_blocking_returns_result_and_records_driver(tmp_path) -> None:
    from imajin.agent import provenance
    from imajin.agent.execution import ToolExecutionService
    from imajin.config import Settings
    from imajin.tools import tool

    provenance.start_session(driver="initial", settings=Settings(data_dir=tmp_path))

    @tool()
    def double(n: int) -> int:
        return n * 2

    service = ToolExecutionService()
    result = service.call_tool_blocking(
        "double",
        {"n": 4},
        source="llm",
        driver="llm:test-model",
    )

    assert result == 8
    [record] = provenance.current_session_path().read_text().strip().splitlines()
    payload = json.loads(record)
    assert payload["driver"] == "llm:test-model"
    assert payload["tool"] == "double"


def test_report_progress_updates_current_job() -> None:
    from imajin.agent.execution import (
        ToolExecutionService,
        current_job_id,
        report_progress,
    )
    from imajin.tools import tool

    @tool()
    def work() -> bool:
        assert current_job_id() is not None
        return report_progress(
            progress=0.4,
            stage="segmentation",
            current_file="sample_1.lsm",
            file_index=1,
            total_files=3,
            completed=0,
            failed=0,
            show_in_chat=True,
            message="Segmenting sample_1.lsm.",
        )

    service = ToolExecutionService()
    job = service.submit_tool("work", {}, source="llm", wait=True)

    assert job.status == "complete"
    assert job.result is True
    assert job.progress == 1.0
    assert job.progress_detail["progress"] == 0.4
    assert job.progress_detail["stage"] == "segmentation"
    assert job.progress_detail["current_file"] == "sample_1.lsm"
    assert job.progress_detail["total_files"] == 3
    assert job.progress_detail["show_in_chat"] is True


def test_validation_error_fails_job() -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import tool

    @tool()
    def needs_int(n: int) -> int:
        return n

    service = ToolExecutionService()
    job = service.submit_tool("needs_int", {"n": "not-an-int"}, wait=True)

    assert job.status == "failed"
    assert "ValidationError" in (job.error or "")


def test_async_worker_tool_completes(qapp, qtbot) -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.tools import tool

    @tool(worker=True)
    def add(a: int, b: int = 1) -> int:
        return a + b

    service = ToolExecutionService()
    job = service.submit_tool("add", {"a": 5, "b": 6}, wait=False)

    qtbot.waitUntil(lambda: job.status == "complete", timeout=2000)
    assert job.result == 11


def test_running_cooperative_job_can_be_cancelled(qapp, qtbot) -> None:
    from imajin.agent.execution import ToolExecutionService, raise_if_cancelled

    service = ToolExecutionService()

    def long_running() -> str:
        while True:
            raise_if_cancelled()
            time.sleep(0.005)

    job = service.submit_workflow(
        "long_running",
        long_running,
        source="manual",
        wait=False,
    )
    qtbot.waitUntil(lambda: job.status == "running", timeout=2000)

    service.cancel(job.job_id)

    qtbot.waitUntil(lambda: job.status == "cancelled", timeout=2000)
    assert "cancel" in (job.message or "").lower()
