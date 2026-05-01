from __future__ import annotations

from typing import Any

import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import NoScrollComboBox, apply_dock_theme


class _DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()

    def setDataFrame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        try:
            value = self._df.iat[index.row(), index.column()]
        except IndexError:
            return None
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except IndexError:
                return None
        return str(section)


class TableDock(QWidget):
    def __init__(self, viewer: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box = QGroupBox("Source")
        source_layout = QHBoxLayout(source_box)
        self.table_picker = NoScrollComboBox()
        source_layout.addWidget(self.table_picker, stretch=1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_table_list)
        source_layout.addWidget(self.refresh_btn)
        layout.addWidget(source_box)

        data_box = QGroupBox("Data")
        data_layout = QVBoxLayout(data_box)
        self._model = _DataFrameModel()
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        data_layout.addWidget(self._view)
        layout.addWidget(data_box, stretch=1)

        self.table_picker.currentTextChanged.connect(self._on_table_change)
        self._view.clicked.connect(self._on_row_click)

        from imajin.agent import state

        state.on_tables_changed(self._refresh_table_list)
        self._refresh_table_list()

    def _refresh_table_list(self) -> None:
        from imajin.agent import state

        prev = self.table_picker.currentText()
        self.table_picker.blockSignals(True)
        self.table_picker.clear()
        for name in state.list_tables():
            self.table_picker.addItem(name)
        if prev and self.table_picker.findText(prev) >= 0:
            self.table_picker.setCurrentText(prev)
        elif self.table_picker.count() > 0:
            self.table_picker.setCurrentIndex(self.table_picker.count() - 1)
        self.table_picker.blockSignals(False)
        self._on_table_change(self.table_picker.currentText())

    def _on_table_change(self, name: str) -> None:
        from imajin.agent import state

        if not name or name not in state.list_tables():
            self._model.setDataFrame(pd.DataFrame())
            return
        self._model.setDataFrame(state.get_table(name))

    def _on_row_click(self, index: QModelIndex) -> None:
        from imajin.agent import state

        if not index.isValid() or self.viewer is None:
            return
        df = state.get_table(self.table_picker.currentText())
        if "label" not in df.columns:
            return
        try:
            label_id = int(df.iloc[index.row()]["label"])
        except (KeyError, ValueError):
            return

        labels_spec = state.get_table_entry(self.table_picker.currentText()).spec.get(
            "labels_layer"
        )
        if not labels_spec or labels_spec not in [L.name for L in self.viewer.layers]:
            return
        layer = self.viewer.layers[labels_spec]
        try:
            layer.selected_label = label_id
            layer.show_selected_label = True
        except Exception:
            pass
