from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imajin.agent.execution import Job, get_execution_service
from imajin.ui.theme import apply_dock_theme


class JobDock(QWidget):
    _refresh_requested = Signal()

    def __init__(self, execution_service: Any | None = None) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.execution_service = execution_service or get_execution_service()
        self._jobs: list[Job] = []
        self._listener = lambda _job: self._refresh_requested.emit()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.cancel_btn = QPushButton("Cancel Selected")
        self.cancel_btn.clicked.connect(self._cancel_selected)
        controls.addWidget(self.cancel_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        table_box = QGroupBox("Jobs")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Status", "Source", "Title", "Progress", "Message"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_details)
        table_layout.addWidget(self.table)
        layout.addWidget(table_box, stretch=2)

        detail_box = QGroupBox("Details")
        detail_layout = QVBoxLayout(detail_box)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(160)
        detail_layout.addWidget(self.details)
        layout.addWidget(detail_box)

        self._refresh_requested.connect(self.refresh)
        self.execution_service.add_listener(self._listener)
        self.refresh()

    def refresh(self) -> None:
        self._jobs = self.execution_service.list_jobs()
        selected_id = self._selected_job_id()
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            values = [
                job.status,
                job.source,
                job.title,
                "" if job.progress is None else f"{job.progress:.0%}",
                job.message or job.error or "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, job.job_id)
                self.table.setItem(row, col, item)
            if job.job_id == selected_id:
                self.table.selectRow(row)
        self.table.resizeColumnsToContents()
        self._update_details()

    def _selected_job_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _selected_job(self) -> Job | None:
        job_id = self._selected_job_id()
        if not job_id:
            return None
        for job in self._jobs:
            if job.job_id == job_id:
                return job
        return None

    def _cancel_selected(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        self.execution_service.cancel(job.job_id)

    def _update_details(self) -> None:
        job = self._selected_job()
        if job is None:
            self.details.clear()
            return
        lines = [
            f"Job: {job.job_id}",
            f"Status: {job.status}",
            f"Source: {job.source}",
            f"Title: {job.title}",
        ]
        if job.tool_name:
            lines.append(f"Tool: {job.tool_name}")
        if job.workflow_name:
            lines.append(f"Workflow: {job.workflow_name}")
        if job.started_at:
            lines.append(f"Started: {job.started_at}")
        if job.finished_at:
            lines.append(f"Finished: {job.finished_at}")
        if job.error:
            lines.append(f"Error: {job.error}")
        if job.result is not None:
            result = repr(job.result)
            if len(result) > 3000:
                result = result[:3000] + "\n... (truncated)"
            lines.append("")
            lines.append(result)
        self.details.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:
        # The listener is intentionally tiny and signal-only; leaving it behind
        # would be harmless, but removing QObject-bound listeners avoids stale
        # signal emissions in tests.
        self.execution_service.remove_listener(self._listener)
        super().closeEvent(event)
