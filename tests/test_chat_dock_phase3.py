from __future__ import annotations


def test_chat_dock_has_model_picker(qtbot, viewer) -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    assert dock.model_picker.count() == 6
    assert "Claude Sonnet 4.6" in dock.model_picker.itemText(0)
    assert "subscription" in dock.model_picker.itemText(2)
    assert "qwen3.5:9b" in dock.model_picker.itemText(5)
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


def test_chat_dock_tool_caller_accepts_tool_input_named_name(qtbot, viewer) -> None:
    from collections.abc import Iterator
    from typing import Any

    from imajin import session as state
    from imajin.agent.providers.base import Event
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    class _NoopProvider:
        name = "noop"
        model = "noop-model"

        def stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            system: str,
        ) -> Iterator[Event]:
            yield from ()

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)
    dock._make_provider = lambda: _NoopProvider()  # type: ignore[method-assign]

    runner = dock._ensure_runner()
    result = runner._tool_caller(
        "create_analysis_recipe",
        name="batch_recipe",
        target_channel="green",
    )

    assert result["recipe_id"] == "batch_recipe"
    assert state.get_recipe("batch_recipe").target_channel == "green"


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


def test_composer_defaults_to_two_visible_lines(qtbot) -> None:
    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput()
    qtbot.addWidget(composer)
    composer.show()
    qtbot.waitUntil(lambda: composer.height() > 0, timeout=1000)

    line_h = composer.fontMetrics().lineSpacing()
    assert composer.height() >= line_h * 2


def test_composer_arrow_keys_move_inside_multiline_text(qtbot) -> None:
    from qtpy.QtCore import Qt

    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput()
    qtbot.addWidget(composer)
    composer.show()
    composer.setPlainText("first line\nsecond line\nthird line")
    cursor = composer.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    composer.setTextCursor(cursor)
    end_pos = composer.textCursor().position()

    qtbot.keyClick(composer, Qt.Key.Key_Up)

    assert composer.textCursor().position() < end_pos


def test_composer_scrollbar_activates_for_many_lines(qtbot) -> None:
    from imajin.ui.chat_dock import _ComposerInput

    composer = _ComposerInput(max_visible_lines=4)
    qtbot.addWidget(composer)
    composer.show()
    composer.setPlainText("\n".join(f"line {i}" for i in range(20)))
    qtbot.waitUntil(lambda: composer.verticalScrollBar().maximum() > 0, timeout=1000)

    assert composer.verticalScrollBar().maximum() > 0


def test_model_picker_text_has_extra_left_padding() -> None:
    from imajin.ui.theme import Theme

    assert "padding: 3px 10px 3px 18px;" in Theme.get_dock_stylesheet()


def test_chat_dock_updates_single_batch_progress_card(qtbot, viewer) -> None:
    from imajin.agent.execution import Job
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    job = Job(
        job_id="job_progress",
        title="run_recipe_on_samples",
        source="llm",
        status="running",
        progress=0.25,
        progress_detail={
            "show_in_chat": True,
            "stage": "segmentation",
            "current_file": r"C:\data\sample_1.lsm",
            "file_index": 1,
            "total_files": 4,
            "completed": 0,
            "failed": 0,
        },
    )
    dock._on_job_update(job)

    job.progress = 0.5
    job.progress_detail.update(
        {
            "stage": "measurement",
            "current_file": r"C:\data\sample_2.lsm",
            "completed": 1,
        }
    )
    dock._on_job_update(job)

    text = dock.transcript.toPlainText()
    assert text.count("Batch progress") == 1
    assert "1/4 files processed" in text
    assert "sample_2.lsm" in text
    assert "measurement" in text
