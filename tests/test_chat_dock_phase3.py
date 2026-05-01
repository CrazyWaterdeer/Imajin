from __future__ import annotations


def test_chat_dock_has_model_picker(qtbot, viewer) -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    assert dock.model_picker.count() == 4
    assert "Claude Sonnet 4.6" in dock.model_picker.itemText(0)
    assert "qwen3.5:9b" in dock.model_picker.itemText(3)
    assert dock.send_btn.isEnabled()
    assert dock.stop_btn.isHidden()


def test_chat_dock_clear_resets_transcript(qtbot, viewer) -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    dock.transcript.append("hello world")
    dock.clear_btn.click()
    assert dock.transcript.toPlainText().strip() == ""
