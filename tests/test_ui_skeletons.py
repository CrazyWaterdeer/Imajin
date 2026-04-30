from __future__ import annotations


def test_chat_dock_instantiates(qtbot) -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=None, settings=Settings())
    qtbot.addWidget(dock)
    assert dock.input.placeholderText() != ""
    assert dock.model_picker.count() > 0


def test_manual_dock_instantiates_without_viewer(qtbot) -> None:
    from imajin.ui.manual_dock import ManualDock

    dock = ManualDock(viewer=None)
    qtbot.addWidget(dock)
    # Tools register at import; picker should be populated regardless of viewer.
    assert dock.tool_picker.count() > 0
