from __future__ import annotations


def test_chat_dock_has_model_picker(qtbot, viewer) -> None:
    from imajin.config import Settings
    from imajin.ui.chat_dock import ChatDock

    dock = ChatDock(viewer=viewer, settings=Settings())
    qtbot.addWidget(dock)

    assert dock.model_picker.count() == 6
    assert "Sonnet" in dock.model_picker.itemText(0)
    assert "latest" in dock.model_picker.itemText(0)
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


def _all_available():
    from imajin.ui.provider_status import ProviderStatus

    return {k: ProviderStatus(True, None) for k in ("anthropic", "claude-agent", "openai", "ollama")}


def test_model_picker_restores_preferred_when_available(qtbot) -> None:
    from imajin.ui.chat_dock import _MODEL_CHOICES, _ModelPickerButton

    btn = _ModelPickerButton(
        _MODEL_CHOICES, statuses=_all_available(), preferred=("claude-agent", "opus")
    )
    qtbot.addWidget(btn)

    _, kind, model = _MODEL_CHOICES[btn.currentIndex()]
    assert (kind, model) == ("claude-agent", "opus")


def test_model_picker_falls_back_when_preferred_unavailable(qtbot) -> None:
    from imajin.ui.chat_dock import _MODEL_CHOICES, _ModelPickerButton
    from imajin.ui.provider_status import ProviderStatus

    statuses = {
        "anthropic": ProviderStatus(True, None),
        "claude-agent": ProviderStatus(False, "no login"),
        "openai": ProviderStatus(False, "no API key"),
        "ollama": ProviderStatus(False, "down"),
    }
    btn = _ModelPickerButton(_MODEL_CHOICES, statuses=statuses, preferred=("openai", "gpt"))
    qtbot.addWidget(btn)

    # preferred (openai) is unavailable -> first available choice (anthropic sonnet, idx 0)
    assert btn.currentIndex() == 0


def test_model_picker_user_click_emits_userSelected(qtbot) -> None:
    from imajin.ui.chat_dock import _MODEL_CHOICES, _ModelPickerButton

    btn = _ModelPickerButton(
        _MODEL_CHOICES, statuses=_all_available(), preferred=("anthropic", "sonnet")
    )
    qtbot.addWidget(btn)

    with qtbot.waitSignal(btn.userSelected, timeout=500) as blocker:
        btn.setCurrentIndex(3)
    assert blocker.args == [3]


def test_model_picker_auto_fallback_does_not_emit_userSelected(qtbot) -> None:
    from imajin.ui.chat_dock import _MODEL_CHOICES, _ModelPickerButton
    from imajin.ui.provider_status import ProviderStatus

    btn = _ModelPickerButton(
        _MODEL_CHOICES, statuses=_all_available(), preferred=("openai", "gpt")
    )
    qtbot.addWidget(btn)
    assert _MODEL_CHOICES[btn.currentIndex()][1] == "openai"

    user_events: list[int] = []
    changed_events: list[int] = []
    btn.userSelected.connect(user_events.append)
    btn.currentIndexChanged.connect(changed_events.append)

    # openai goes unavailable -> auto-fallback, must not be persisted as a user choice
    statuses = _all_available()
    statuses["openai"] = ProviderStatus(False, "no API key")
    btn.refresh_statuses(statuses)

    assert btn.currentIndex() == 0  # fell back to first available
    assert changed_events == [0]  # a change did happen
    assert user_events == []  # but it was not a user selection


def test_chat_dock_persists_and_restores_model_choice(qtbot, viewer, tmp_path, monkeypatch) -> None:
    import json
    from unittest.mock import patch

    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "compute_statuses", lambda _s: _all_available())
    secrets = tmp_path / "secrets.json"

    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        settings = Settings()  # defaults: anthropic / sonnet -> index 0
        dock = cd.ChatDock(viewer=viewer, settings=settings)
        qtbot.addWidget(dock)
        assert dock.model_picker.currentIndex() == 0

        target = 3  # ("claude-agent", "opus")
        dock.model_picker.setCurrentIndex(target)  # simulate a user pick

        # in-memory + on-disk both updated
        assert (settings.default_provider, settings.default_model) == ("claude-agent", "opus")
        raw = json.loads(secrets.read_text())
        assert (raw["default_provider"], raw["default_model"]) == ("claude-agent", "opus")

        # a fresh load (restart) restores the same selection
        restored = Settings.from_env()
        assert (restored.default_provider, restored.default_model) == ("claude-agent", "opus")
        dock2 = cd.ChatDock(viewer=viewer, settings=restored)
        qtbot.addWidget(dock2)
        assert dock2.model_picker.currentIndex() == target
