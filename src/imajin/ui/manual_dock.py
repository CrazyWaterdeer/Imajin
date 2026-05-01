from __future__ import annotations

import inspect
from typing import Any

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
    return {p for p in sig.parameters if p == "layer" or p.endswith("_layer")}


class ManualDock(QWidget):
    def __init__(self, viewer: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer
        self._current_widget = None

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
        from imajin.tools import iter_tools

        self.tool_picker.clear()
        entries = sorted(iter_tools(), key=lambda e: (e.phase, e.name))
        for entry in entries:
            self.tool_picker.addItem(f"[{entry.phase}] {entry.name}", entry)

    def _layer_choices(self, _widget) -> list[str]:
        if self.viewer is None:
            return []
        return [L.name for L in self.viewer.layers]

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
        overrides = {p: {"choices": self._layer_choices} for p in layer_params}

        try:
            gui = magicgui(entry.func, call_button="Run", **overrides)
        except Exception as e:
            self.result_view.setPlainText(
                f"Could not build form for {entry.name}: {e}"
            )
            return

        gui.called.connect(self._on_result)
        self._current_widget = gui
        self._form_layout.addWidget(gui.native)

    def _on_result(self, result: Any) -> None:
        try:
            text = repr(result)
            if len(text) > 4000:
                text = text[:4000] + "\n... (truncated)"
        except Exception as e:
            text = f"<unrepresentable result: {e}>"
        self.result_view.setPlainText(text)
