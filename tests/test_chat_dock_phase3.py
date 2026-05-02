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


def test_composer_accepts_korean_input_method_text(qtbot) -> None:
    from qtpy.QtGui import QInputMethodEvent
    from qtpy.QtWidgets import QApplication

    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput()
    qtbot.addWidget(composer)

    event = QInputMethodEvent()
    event.setCommitString("안녕하세요")
    QApplication.sendEvent(composer, event)

    assert composer.toPlainText() == "안녕하세요"


def test_composer_enter_does_not_submit_during_ime_preedit(qtbot) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QInputMethodEvent
    from qtpy.QtWidgets import QApplication

    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput()
    qtbot.addWidget(composer)
    submitted: list[bool] = []
    composer.submitted.connect(lambda: submitted.append(True))

    QApplication.sendEvent(composer, QInputMethodEvent("ㅎ", []))
    qtbot.keyClick(composer, Qt.Key.Key_Return)

    assert submitted == []


def test_composer_accepts_shortcut_override_while_focused(qtbot) -> None:
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication

    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput()
    qtbot.addWidget(composer)
    composer.show()
    composer.setFocus()
    qtbot.waitUntil(lambda: composer.hasFocus(), timeout=1000)

    event = QKeyEvent(
        QEvent.Type.ShortcutOverride,
        Qt.Key.Key_G,
        Qt.KeyboardModifier.NoModifier,
        "g",
    )
    QApplication.sendEvent(composer, event)

    assert event.isAccepted()
