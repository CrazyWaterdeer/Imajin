from __future__ import annotations

import functools
import inspect
from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import NoScrollComboBox, apply_dock_theme


def _layer_param_names(func) -> set[str]:
    sig = inspect.signature(func)
    names = set()
    for p in sig.parameters:
        if p == "layer" or p.endswith("_layer"):
            names.add(p)
        elif p in {"image_a", "image_b", "mask"}:
            names.add(p)
    return names


def _table_param_names(func) -> set[str]:
    sig = inspect.signature(func)
    return {p for p in sig.parameters if p == "table_name"}


def _manual_callable(entry, execution_service):
    @functools.wraps(entry.func)
    def wrapped(*args, **kwargs):
        job = execution_service.submit_tool(
            entry.name,
            kwargs=dict(kwargs),
            source="manual",
            title=entry.name,
        )
        return {
            "job_id": job.job_id,
            "tool": entry.name,
            "status": job.status,
            "message": job.message,
        }

    wrapped.__signature__ = inspect.signature(entry.func)  # type: ignore[attr-defined]
    return wrapped


class ManualDock(QWidget):
    _job_updated = Signal(object)

    def __init__(self, viewer: Any, execution_service: Any | None = None) -> None:
        super().__init__()
        apply_dock_theme(self)
        from imajin.agent.execution import get_execution_service

        self.viewer = viewer
        self.execution_service = execution_service or get_execution_service()
        self._current_widget = None
        self._last_job_id: str | None = None
        self._job_updated.connect(self._on_job_updated)
        self.execution_service.add_listener(self._emit_job_update)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        tool_box = QGroupBox("Tool")
        tool_layout = QHBoxLayout(tool_box)
        self.tool_picker = NoScrollComboBox()
        tool_layout.addWidget(self.tool_picker)
        layout.addWidget(tool_box)

        params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_box)
        self._form_holder = QWidget()
        self._form_layout = QVBoxLayout(self._form_holder)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(self._form_holder)
        layout.addWidget(params_box, stretch=1)

        output_box = QGroupBox("Output")
        output_layout = QVBoxLayout(output_box)
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMaximumHeight(160)
        self.result_view.setPlaceholderText("Tool output appears here.")
        output_layout.addWidget(self.result_view)
        layout.addWidget(output_box)

        self._populate_tools()
        self.tool_picker.currentIndexChanged.connect(self._on_tool_change)
        if self.tool_picker.count() > 0:
            self._on_tool_change(0)

    def _populate_tools(self) -> None:
        from imajin.tools import manual_tools

        self.tool_picker.clear()
        entries = sorted(manual_tools(), key=lambda e: (e.phase, e.name))
        for entry in entries:
            self.tool_picker.addItem(f"[{entry.phase}] {entry.name}", entry)

    def _layer_choices(self, _widget) -> list[str]:
        if self.viewer is None:
            return []
        return [L.name for L in self.viewer.layers]

    def _table_choices(self, _widget) -> list[str]:
        from imajin.agent import state

        return state.list_tables()

    def _on_tool_change(self, index: int) -> None:
        from magicgui import magicgui

        if self._current_widget is not None:
            self._form_layout.removeWidget(self._current_widget.native)
            self._current_widget.native.deleteLater()
            self._current_widget = None

        if index < 0 or self.tool_picker.count() == 0:
            return
        entry = self.tool_picker.itemData(index)
        if entry is None:
            return

        layer_params = _layer_param_names(entry.func)
        table_params = _table_param_names(entry.func)
        overrides = {p: {"choices": self._layer_choices} for p in layer_params}
        overrides.update({p: {"choices": self._table_choices} for p in table_params})

        try:
            gui = magicgui(
                _manual_callable(entry, self.execution_service),
                call_button="Run",
                **overrides,
            )
        except Exception as e:
            self.result_view.setPlainText(
                f"Could not build form for {entry.name}: {e}"
            )
            return

        gui.called.connect(self._on_result)
        self._current_widget = gui
        self._form_layout.addWidget(gui.native)

    def _on_result(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("job_id"):
            self._last_job_id = str(result["job_id"])
        try:
            text = repr(result)
            if len(text) > 4000:
                text = text[:4000] + "\n... (truncated)"
        except Exception as e:
            text = f"<unrepresentable result: {e}>"
        self.result_view.setPlainText(text)

    def _emit_job_update(self, job: Any) -> None:
        self._job_updated.emit(job)

    def _on_job_updated(self, job: Any) -> None:
        if self._last_job_id != getattr(job, "job_id", None):
            return
        if job.status in {"queued", "running", "cancel_requested"}:
            self.result_view.setPlainText(
                f"{job.job_id} {job.status}: {job.title}\n{job.message or ''}"
            )
            return
        if job.status == "failed":
            self.result_view.setPlainText(
                f"{job.job_id} failed: {job.title}\n{job.error or ''}"
            )
            return
        if job.status == "cancelled":
            self.result_view.setPlainText(
                f"{job.job_id} cancelled: {job.title}\n{job.message or ''}"
            )
            return
        text = repr(job.result)
        if len(text) > 4000:
            text = text[:4000] + "\n... (truncated)"
        self.result_view.setPlainText(
            f"{job.job_id} complete: {job.title}\n{text}"
        )

    def closeEvent(self, event) -> None:
        self.execution_service.remove_listener(self._emit_job_update)
        super().closeEvent(event)
