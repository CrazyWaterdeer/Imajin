from __future__ import annotations

import json
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeyEvent, QTextCursor
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import Theme, apply_dock_theme

_MODEL_CHOICES: list[tuple[str, str, str]] = [
    ("Claude Sonnet 4.6", "anthropic", "claude-sonnet-4-6"),
    ("Claude Opus 4.7", "anthropic", "claude-opus-4-7"),
    ("GPT-5 (OpenAI)", "openai", "gpt-5"),
    ("Local: qwen3.5:9b (multimodal, 256K)", "ollama", "qwen3.5:9b"),
]


def _short_label(label: str) -> str:
    short = label.replace("Claude ", "").replace(" (OpenAI)", "").replace("Local: ", "")
    if len(short) > 26:
        short = short[:24] + "…"
    return short


class _ModelPickerButton(QPushButton):
    """Pill-shaped button that opens a menu of model choices."""

    currentIndexChanged = Signal(int)

    def __init__(self, choices: list[tuple[str, str, str]], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelBtn")
        self._choices = choices
        self._index = 0

        menu = QMenu(self)
        last_kind: str | None = None
        for i, (label, kind, _) in enumerate(choices):
            if last_kind is not None and kind != last_kind:
                menu.addSeparator()
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, idx=i: self.setCurrentIndex(idx))
            last_kind = kind
        self.setMenu(menu)
        self._refresh_text()

    def setCurrentIndex(self, idx: int) -> None:
        if idx == self._index:
            return
        self._index = idx
        self._refresh_text()
        self.currentIndexChanged.emit(idx)

    def currentIndex(self) -> int:
        return self._index

    def count(self) -> int:
        return len(self._choices)

    def itemText(self, idx: int) -> str:
        return self._choices[idx][0]

    def _refresh_text(self) -> None:
        label = self._choices[self._index][0]
        self.setText(f"{_short_label(label)}  ▾")


class _ComposerInput(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, max_visible_lines: int = 4) -> None:
        super().__init__()
        self._max_visible_lines = max_visible_lines
        self._frame_padding = 12
        self.setFixedHeight(34)  # provisional; refined once the font is realized
        self.document().contentsChanged.connect(self._adjust_height)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._adjust_height()

    def _adjust_height(self) -> None:
        line_h = self.fontMetrics().lineSpacing()
        if line_h <= 0:
            return
        blocks = max(1, self.document().blockCount())
        visible = min(blocks, self._max_visible_lines)
        target = visible * line_h + self._frame_padding
        self.setFixedHeight(target)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(event)


class ChatDock(QWidget):
    def __init__(self, viewer: Any, settings: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer
        self.settings = settings
        self._runner = None
        self._worker = None
        self._provider_kind = None
        self._provider_model = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText(
            "Type a request below.   e.g. 이 z-stack에서 세포 찾고 채널2 강도 측정해줘"
        )
        layout.addWidget(self.transcript, stretch=1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(8, 6, 8, 6)
        composer_layout.setSpacing(4)

        self.input = _ComposerInput(max_visible_lines=4)
        self.input.setPlaceholderText("Type a request…   (Shift+Enter = newline)")
        self.input.submitted.connect(self._on_send)
        composer_layout.addWidget(self.input)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self.model_picker = _ModelPickerButton(_MODEL_CHOICES)
        self.model_picker.currentIndexChanged.connect(self._on_model_change)
        toolbar.addWidget(self.model_picker)

        toolbar.addStretch(1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("composerTool")
        self.clear_btn.setToolTip("Reset conversation history")
        self.clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self.clear_btn)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setToolTip("Send (Enter)")
        self.send_btn.clicked.connect(self._on_send)
        toolbar.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setToolTip("Cancel the running turn")
        self.stop_btn.clicked.connect(self._on_cancel)
        self.stop_btn.hide()
        toolbar.addWidget(self.stop_btn)

        self.cancel_btn = self.stop_btn  # backward-compat alias

        composer_layout.addLayout(toolbar)
        layout.addWidget(composer)

    def invalidate_runner(self) -> None:
        self._runner = None
        self._provider_kind = None
        self._provider_model = None

    def _make_provider(self):
        from imajin.agent.providers import (
            AnthropicProvider,
            OpenAICompatProvider,
        )

        idx = self.model_picker.currentIndex()
        _, kind, model = _MODEL_CHOICES[idx]
        if kind == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Open Imajin → API Keys… or set the env var."
                )
            return AnthropicProvider(
                api_key=self.settings.anthropic_api_key, model=model
            )
        if kind == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Open Imajin → API Keys… or set the env var."
                )
            return OpenAICompatProvider(
                api_key=self.settings.openai_api_key,
                model=model,
                base_url=self.settings.openai_base_url,
            )
        return OpenAICompatProvider(
            api_key=None, model=model, base_url=self.settings.ollama_base_url
        )

    def _ensure_runner(self):
        from imajin.agent.prompts import build_system_prompt
        from imajin.agent.runner import AgentRunner

        idx = self.model_picker.currentIndex()
        _, kind, model = _MODEL_CHOICES[idx]
        if (
            self._runner is not None
            and self._provider_kind == kind
            and self._provider_model == model
        ):
            return self._runner
        provider = self._make_provider()
        self._runner = AgentRunner(provider, build_system_prompt())
        self._provider_kind = kind
        self._provider_model = model
        return self._runner

    def _on_model_change(self, _index: int) -> None:
        if self._runner is not None:
            self._append_system("Model changed — conversation reset.")
        self._runner = None

    def _on_clear(self) -> None:
        if self._runner is not None:
            self._runner.reset()
        self.transcript.clear()

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()

        try:
            runner = self._ensure_runner()
        except Exception as e:
            self._append_system(f"[error] {e}")
            return

        self._append_user(text)
        self._begin_assistant_turn()

        from napari.qt import thread_worker

        @thread_worker
        def _do_turn():
            for event in runner.turn(text):
                yield event

        worker = _do_turn()
        worker.yielded.connect(self._on_event)
        worker.finished.connect(self._on_finished)
        worker.errored.connect(self._on_errored)
        self._worker = worker
        self._set_streaming(True)
        worker.start()

    def _on_cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass

    def _on_finished(self) -> None:
        self._set_streaming(False)
        self._worker = None

    def _on_errored(self, exc: Exception) -> None:
        self._append_system(f"[runner error] {type(exc).__name__}: {exc}")
        self._on_finished()

    def _set_streaming(self, streaming: bool) -> None:
        self.send_btn.setVisible(not streaming)
        self.stop_btn.setVisible(streaming)

    def _on_event(self, event: Any) -> None:
        from imajin.agent.providers.base import (
            Stop,
            TextDelta,
            ToolUse,
            ToolUseStart,
        )
        from imajin.agent.runner import ToolResult, TurnDone

        if isinstance(event, TextDelta):
            self._append_text_delta(event.text)
        elif isinstance(event, ToolUseStart):
            self._append_system(f"→ {event.name}…")
        elif isinstance(event, ToolUse):
            args = json.dumps(event.input, ensure_ascii=False, default=str)
            if len(args) > 200:
                args = args[:200] + "…"
            self._append_system(f"   args: {args}")
        elif isinstance(event, ToolResult):
            tag = "ERROR" if event.is_error else "ok"
            out = repr(event.output)
            if len(out) > 240:
                out = out[:240] + "…"
            self._append_system(f"   ← {tag}: {out}")
        elif isinstance(event, Stop):
            pass
        elif isinstance(event, TurnDone):
            usage = event.total_usage
            usage_str = ""
            if usage:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                cache_r = usage.get("cache_read_input_tokens", 0)
                usage_str = f" — tokens: in {inp} (cache_read {cache_r}), out {out}"
            self._append_system(f"[turn complete: {event.stop_reason}]{usage_str}")

    def _append_user(self, text: str) -> None:
        self.transcript.append("")
        self.transcript.append(f"<b>User:</b> {self._escape(text)}")

    def _begin_assistant_turn(self) -> None:
        self.transcript.append("<b>Assistant:</b>")

    def _append_text_delta(self, text: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.transcript.setTextCursor(cursor)

    def _append_system(self, msg: str) -> None:
        self.transcript.append(
            f"<span style='color:{Theme.TEXT_SECONDARY}'><i>{self._escape(msg)}</i></span>"
        )

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
