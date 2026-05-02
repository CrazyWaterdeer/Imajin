from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from imajin.workers.qt_worker import CancellationToken, CancelledError

JobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "complete",
    "failed",
]

JobSource = Literal["manual", "llm", "batch", "system"]


@dataclass
class Job:
    job_id: str
    title: str
    source: JobSource
    tool_name: str | None = None
    workflow_name: str | None = None
    status: JobStatus = "queued"
    progress: float | None = None
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: Any | None = None
    error: str | None = None
    provenance_session_id: str | None = None


@dataclass(frozen=True)
class ProgressEvent:
    job_id: str
    progress: float | None = None
    message: str | None = None


_CURRENT_TOKEN: ContextVar[CancellationToken | None] = ContextVar(
    "imajin_current_cancellation_token",
    default=None,
)


def current_cancellation_token() -> CancellationToken | None:
    return _CURRENT_TOKEN.get()


def raise_if_cancelled() -> None:
    token = current_cancellation_token()
    if token is not None:
        token.raise_if_cancelled()


class ToolExecutionService:
    """Central job runner for manual, LLM, batch, and system tool calls."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._listeners: list[Callable[[Job], None]] = []
        self._active_workers: dict[str, Any] = {}
        self._lock = threading.RLock()

    def add_listener(self, callback: Callable[[Job], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Job], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def submit_tool(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
        source: JobSource = "manual",
        title: str | None = None,
        driver: str | None = None,
        wait: bool | None = None,
        tool_caller: Callable[..., Any] | None = None,
    ) -> Job:
        """Submit a tool call and return its job record.

        ``wait=None`` uses the tool metadata: worker-enabled tools run
        asynchronously, while main-thread tools run inline. LLM callers usually
        use ``call_tool_blocking`` so the model receives a concrete result.
        """

        from imajin.tools.registry import get_tool

        kwargs = dict(kwargs or {})
        entry = get_tool(name)
        if wait is None:
            wait = not entry.worker

        job = self._new_job(
            title=title or name,
            source=source,
            tool_name=name,
        )
        token = CancellationToken()
        with self._lock:
            self._tokens[job.job_id] = token

        if wait:
            self._execute_tool_job(
                job=job,
                name=name,
                kwargs=kwargs,
                source=source,
                driver=driver,
                token=token,
                tool_caller=tool_caller,
                raise_errors=False,
            )
            return job

        self._start_async_tool_job(
            job=job,
            name=name,
            kwargs=kwargs,
            source=source,
            driver=driver,
            token=token,
            tool_caller=tool_caller,
        )
        return job

    def call_tool_blocking(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
        source: JobSource = "llm",
        title: str | None = None,
        driver: str | None = None,
        tool_caller: Callable[..., Any] | None = None,
    ) -> Any:
        job = self._new_job(
            title=title or name,
            source=source,
            tool_name=name,
        )
        token = CancellationToken()
        with self._lock:
            self._tokens[job.job_id] = token
        return self._execute_tool_job(
            job=job,
            name=name,
            kwargs=dict(kwargs or {}),
            source=source,
            driver=driver,
            token=token,
            tool_caller=tool_caller,
            raise_errors=True,
        )

    def submit_workflow(
        self,
        workflow_name: str,
        callable_: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        source: JobSource = "manual",
        title: str | None = None,
        driver: str | None = None,
        wait: bool = False,
    ) -> Job:
        job = self._new_job(
            title=title or workflow_name,
            source=source,
            workflow_name=workflow_name,
        )
        token = CancellationToken()
        with self._lock:
            self._tokens[job.job_id] = token

        def runner() -> Any:
            return callable_(**dict(kwargs or {}))

        if wait:
            self._execute_callable_job(job, runner, source, driver, token, False)
            return job

        self._start_async_callable_job(job, runner, source, driver, token)
        return job

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            token = self._tokens.get(job_id)
            if token is not None:
                token.cancel()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = _now()
                job.message = "Cancelled before start."
            elif job.status == "running":
                job.status = "cancel_requested"
                job.message = "Cancellation requested."
            elif job.status == "cancel_requested":
                return
            else:
                return
        self._notify(job)

    def cancel_running(self, source: JobSource | None = None) -> list[str]:
        cancelled: list[str] = []
        with self._lock:
            ids = [
                job.job_id
                for job in self._jobs.values()
                if job.status in {"queued", "running", "cancel_requested"}
                and (source is None or job.source == source)
            ]
        for job_id in ids:
            self.cancel(job_id)
            cancelled.append(job_id)
        return cancelled

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def clear_jobs(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._tokens.clear()
            self._active_workers.clear()

    def replace_jobs(self, jobs: list[Job]) -> None:
        with self._lock:
            self._jobs = {job.job_id: job for job in jobs}
            self._tokens.clear()
            self._active_workers.clear()
        for job in jobs:
            self._notify(job)

    def _new_job(
        self,
        *,
        title: str,
        source: JobSource,
        tool_name: str | None = None,
        workflow_name: str | None = None,
    ) -> Job:
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:10]}",
            title=title,
            source=source,
            tool_name=tool_name,
            workflow_name=workflow_name,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._notify(job)
        return job

    def _start_async_tool_job(
        self,
        *,
        job: Job,
        name: str,
        kwargs: dict[str, Any],
        source: JobSource,
        driver: str | None,
        token: CancellationToken,
        tool_caller: Callable[..., Any] | None,
    ) -> None:
        def runner() -> Any:
            return self._execute_tool_job(
                job=job,
                name=name,
                kwargs=kwargs,
                source=source,
                driver=driver,
                token=token,
                tool_caller=tool_caller,
                raise_errors=False,
            )

        self._start_worker(job, runner)

    def _start_async_callable_job(
        self,
        job: Job,
        callable_: Callable[[], Any],
        source: JobSource,
        driver: str | None,
        token: CancellationToken,
    ) -> None:
        def runner() -> Any:
            return self._execute_callable_job(
                job=job,
                callable_=callable_,
                source=source,
                driver=driver,
                token=token,
                raise_errors=False,
            )

        self._start_worker(job, runner)

    def _start_worker(self, job: Job, runner: Callable[[], Any]) -> None:
        try:
            from napari.qt import thread_worker

            @thread_worker
            def _run():
                return runner()

            worker = _run()
            worker.finished.connect(lambda: self._active_workers.pop(job.job_id, None))
            with self._lock:
                self._active_workers[job.job_id] = worker
            worker.start()
        except Exception:
            def _thread_runner() -> None:
                try:
                    runner()
                finally:
                    self._active_workers.pop(job.job_id, None)

            thread = threading.Thread(target=_thread_runner, daemon=True)
            with self._lock:
                self._active_workers[job.job_id] = thread
            thread.start()

    def _execute_tool_job(
        self,
        *,
        job: Job,
        name: str,
        kwargs: dict[str, Any],
        source: JobSource,
        driver: str | None,
        token: CancellationToken,
        tool_caller: Callable[..., Any] | None,
        raise_errors: bool,
    ) -> Any:
        try:
            from imajin.tools.registry import get_tool

            entry = get_tool(name)
            validated = entry.input_model(**kwargs).model_dump()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = "failed"
                job.finished_at = _now()
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed during input validation."
            self._notify(job)
            if raise_errors:
                raise
            return None

        def run() -> Any:
            from imajin.tools import call_tool

            caller = tool_caller or call_tool
            return caller(name, **validated)

        return self._execute_callable_job(
            job=job,
            callable_=run,
            source=source,
            driver=driver,
            token=token,
            raise_errors=raise_errors,
        )

    def _execute_callable_job(
        self,
        job: Job,
        callable_: Callable[[], Any],
        source: JobSource,
        driver: str | None,
        token: CancellationToken,
        raise_errors: bool,
    ) -> Any:
        from imajin.agent import provenance

        if token.is_cancelled():
            with self._lock:
                job.status = "cancelled"
                job.finished_at = _now()
                job.message = "Cancelled before start."
            self._notify(job)
            if raise_errors:
                raise CancelledError("Tool execution cancelled by user.")
            return None

        with self._lock:
            job.status = "running"
            job.started_at = _now()
            job.message = "Running."
        self._notify(job)

        token_var = _CURRENT_TOKEN.set(token)
        try:
            token.raise_if_cancelled()
            provenance.set_driver(driver or _driver_for_source(source))
            job.provenance_session_id = provenance.current_session_id()
            result = callable_()
            if token.is_cancelled():
                with self._lock:
                    job.status = "cancelled"
                    job.finished_at = _now()
                    job.message = "Cancelled; result ignored."
                self._notify(job)
                if raise_errors:
                    raise CancelledError("Tool execution cancelled by user.")
                return None
            with self._lock:
                job.status = "complete"
                job.finished_at = _now()
                job.result = result
                job.error = None
                job.progress = 1.0
                job.message = "Complete."
                job.provenance_session_id = provenance.current_session_id()
            self._notify(job)
            return result
        except CancelledError as exc:
            with self._lock:
                job.status = "cancelled"
                job.finished_at = _now()
                job.error = str(exc)
                job.message = "Cancelled."
            self._notify(job)
            if raise_errors:
                raise
            return None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = "failed"
                job.finished_at = _now()
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed."
            self._notify(job)
            if raise_errors:
                raise
            return None
        finally:
            _CURRENT_TOKEN.reset(token_var)

    def _notify(self, job: Job) -> None:
        for listener in list(self._listeners):
            try:
                listener(job)
            except Exception:
                pass
        if job.status in {"complete", "failed", "cancelled"}:
            try:
                from imajin.project import autosave_current_project

                autosave_current_project(f"job_{job.status}")
            except Exception:
                pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _driver_for_source(source: JobSource) -> str:
    if source == "llm":
        return "llm"
    return source


_DEFAULT_SERVICE: ToolExecutionService | None = None


def get_execution_service() -> ToolExecutionService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = ToolExecutionService()
    return _DEFAULT_SERVICE
