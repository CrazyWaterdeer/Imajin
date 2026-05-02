from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import NoScrollComboBox, apply_dock_theme


class QCDock(QWidget):
    def __init__(self, viewer: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box = QGroupBox("Source")
        source_layout = QHBoxLayout(source_box)
        self.source_picker = NoScrollComboBox()
        self.source_picker.currentIndexChanged.connect(self._on_source_change)
        source_layout.addWidget(self.source_picker, stretch=1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        source_layout.addWidget(self.refresh_btn)
        layout.addWidget(source_box)

        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("not_checked")
        status_layout.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute_qc)
        actions.addWidget(self.compute_btn)
        for label, status in (
            ("Pass", "pass"),
            ("Warn", "warning"),
            ("Fail", "fail"),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, s=status: self._mark(s))
            actions.addWidget(btn)
        status_layout.addLayout(actions)
        layout.addWidget(status_box)

        warnings_box = QGroupBox("Warnings")
        warnings_layout = QVBoxLayout(warnings_box)
        self.warnings_view = QTextEdit()
        self.warnings_view.setReadOnly(True)
        self.warnings_view.setMaximumHeight(120)
        warnings_layout.addWidget(self.warnings_view)
        layout.addWidget(warnings_box)

        metrics_box = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)
        layout.addWidget(metrics_box, stretch=1)

        notes_box = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_box)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_edit)
        layout.addWidget(notes_box)

        self.refresh()

    def refresh(self) -> None:
        from imajin.agent import state

        previous = self.source_picker.currentData()
        items: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        for record in state.list_qc_records():
            source = str(record.get("source") or "")
            if not source:
                continue
            kind = self._infer_kind(source)
            key = (kind, source)
            if key not in seen:
                items.append((f"{kind}: {source}", kind, source))
                seen.add(key)

        for table in state.list_tables():
            key = ("table", table)
            if key not in seen:
                items.append((f"table: {table}", "table", table))
                seen.add(key)

        for layer in getattr(self.viewer, "layers", []):
            if self._is_labels_layer(layer):
                key = ("labels", layer.name)
                if key not in seen:
                    items.append((f"labels: {layer.name}", "labels", layer.name))
                    seen.add(key)

        self.source_picker.blockSignals(True)
        self.source_picker.clear()
        for label, kind, source in items:
            self.source_picker.addItem(label, (kind, source))
        if previous is not None:
            try:
                prev_kind, prev_source = previous
            except (TypeError, ValueError):
                prev_kind, prev_source = "", ""
            idx = self._find_source_index(str(prev_kind), str(prev_source))
            if idx >= 0:
                self.source_picker.setCurrentIndex(idx)
        self.source_picker.blockSignals(False)
        self._on_source_change(self.source_picker.currentIndex())

    def _on_source_change(self, _index: int) -> None:
        from imajin.agent import state

        current = self._current_source()
        if current is None:
            self.status_label.setText("not_checked")
            self.warnings_view.clear()
            self.metrics_table.setRowCount(0)
            self.notes_edit.clear()
            return
        _kind, source = current
        try:
            record = state.get_qc_record(source)
        except KeyError:
            self.status_label.setText("not_checked")
            self.warnings_view.clear()
            self.metrics_table.setRowCount(0)
            self.notes_edit.clear()
            return

        self.status_label.setText(record.status)
        self.warnings_view.setPlainText("\n".join(record.warnings))
        self.notes_edit.setPlainText(record.notes or "")
        self._set_metrics(record.metrics)

    def _set_metrics(self, metrics: dict[str, Any]) -> None:
        rows = sorted(metrics.items(), key=lambda item: item[0])
        self.metrics_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            text = repr(value)
            if len(text) > 300:
                text = text[:297] + "..."
            self.metrics_table.setItem(row, 1, QTableWidgetItem(text))
        self.metrics_table.resizeColumnsToContents()

    def _current_source(self) -> tuple[str, str] | None:
        data = self.source_picker.currentData()
        if not data:
            return None
        kind, source = data
        return str(kind), str(source)

    def _find_source_index(self, kind: str, source: str) -> int:
        for i in range(self.source_picker.count()):
            data = self.source_picker.itemData(i)
            if not data:
                continue
            item_kind, item_source = data
            if str(item_kind) == kind and str(item_source) == source:
                return i
        return -1

    def _compute_qc(self) -> None:
        from imajin.agent import state
        from imajin.tools import qc

        current = self._current_source()
        if current is None:
            return
        kind, source = current
        try:
            if kind == "labels":
                qc.compute_segmentation_qc(source)
            elif kind == "table":
                entry = state.get_table_entry(source)
                if entry.spec.get("tool") == "measure_intensity_over_time":
                    qc.compute_timecourse_qc(source)
                else:
                    qc.compute_measurement_qc(source)
            else:
                return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "QC", str(exc))
            return
        self.refresh()

    def _mark(self, status: str) -> None:
        from imajin.tools import qc

        current = self._current_source()
        if current is None:
            return
        _kind, source = current
        notes = self.notes_edit.toPlainText().strip() or None
        qc.mark_qc_status(source, status, notes=notes)
        self.refresh()

    def _infer_kind(self, source: str) -> str:
        from imajin.agent import state

        if source in state.list_tables():
            return "table"
        for layer in getattr(self.viewer, "layers", []):
            if getattr(layer, "name", None) == source and self._is_labels_layer(layer):
                return "labels"
        return "record"

    @staticmethod
    def _is_labels_layer(layer: Any) -> bool:
        if getattr(layer, "kind", None) == "labels":
            return True
        class_name = type(layer).__name__.lower()
        return "label" in class_name and hasattr(layer, "selected_label")
