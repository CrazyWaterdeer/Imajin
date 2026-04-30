from __future__ import annotations


def test_manual_dock_populates_tools(qtbot, viewer) -> None:
    from imajin.ui.manual_dock import ManualDock

    dock = ManualDock(viewer=viewer)
    qtbot.addWidget(dock)

    assert dock.tool_picker.count() >= 6, "expected at least 6 registered tools"
    assert dock._current_widget is not None, "form should be built for first tool"


def test_manual_dock_form_rebuilds_on_change(qtbot, viewer) -> None:
    from imajin.ui.manual_dock import ManualDock

    dock = ManualDock(viewer=viewer)
    qtbot.addWidget(dock)

    first = dock._current_widget
    if dock.tool_picker.count() >= 2:
        dock.tool_picker.setCurrentIndex(1)
        assert dock._current_widget is not first
