from __future__ import annotations

import json
from typing import Any

from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import NoScrollComboBox, Theme, apply_dock_theme

_MODEL_CHOICES: list[tuple[str, str, str]] = [
    ("Claude Sonnet 4.6", "anthropic", "claude-sonnet-4-6"),
    ("Claude Opus 4.7", "anthropic", "claude-opus-4-7"),
    ("GPT-5 (OpenAI)", "openai", "gpt-5"),
    ("Local: qwen3.5:9b (multimodal, 256K)", "ollama", "qwen3.5:9b"),
]


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

        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Model:</b>"))
        self.model_picker = NoScrollComboBox()
        for label, _, _ in _MODEL_CHOICES:
            self.model_picker.addItem(label)
        head.addWidget(self.model_picker, stretch=1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Reset conversation history")
        self.clear_btn.clicked.connect(self._on_clear)
        head.addWidget(self.clear_btn)
        layout.addLayout(head)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText(
            "Type a request below. e.g. 이 z-stack에서 세포 찾고 채널2 강도 측정해줘"
        )
        layout.addWidget(self.transcript, stretch=1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Send a message...")
        self.input.returnPressed.connect(self._on_send)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self.input, stretch=1)
        row.addWidget(self.send_btn)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

        self.model_picker.currentIndexChanged.connect(self._on_model_change)

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
                    "ANTHROPIC_API_KEY not set. Set the env var or use a local provider."
                )
            return AnthropicProvider(
                api_key=self.settings.anthropic_api_key, model=model
            )
        if kind == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set the env var or use a local provider."
                )
            return OpenAICompatProvider(
                api_key=self.settings.openai_api_key,
                model=model,
                base_url="https://api.openai.com/v1",
            )
        return OpenAICompatProvider(
            api_key=None, model=model, base_url="http://localhost:11434/v1"
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
        text = self.input.text().strip()
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
        self.send_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
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
        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._worker = None

    def _on_errored(self, exc: Exception) -> None:
        self._append_system(f"[runner error] {type(exc).__name__}: {exc}")
        self._on_finished()

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
