"""Chat transcript with role-aware message bubbles.

Replaces the flat QTextEdit transcript. User messages are right-aligned
with an accent bubble; assistant messages are left-aligned with a subtle
bubble; system/tool messages are centered, muted, and small.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MessageBubble(QFrame):
    """A single message rendered as a styled frame containing a wrapped label."""

    def __init__(self, role: str, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.role = role
        self.setObjectName(
            {
                "user": "bubbleUser",
                "assistant": "bubbleAssistant",
                "system": "bubbleSystem",
            }.get(role, "bubbleAssistant")
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.label)

    def text(self) -> str:
        return self.label.text()

    def append_text(self, delta: str) -> None:
        self.label.setText(self.label.text() + delta)


class ChatTranscript(QScrollArea):
    """Scrollable column of MessageBubble widgets."""

    BUBBLE_MAX_RATIO = 0.78  # cap bubble width to ~78% of viewport

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("chatTranscript")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._inner = QWidget()
        self._inner.setObjectName("chatTranscriptInner")
        self.setWidget(self._inner)

        self._vbox = QVBoxLayout(self._inner)
        self._vbox.setContentsMargins(8, 8, 8, 8)
        self._vbox.setSpacing(8)
        # Trailing stretch so bubbles stack from the top.
        self._vbox.addStretch(1)

        self._bubbles: list[MessageBubble] = []
        self._current_assistant: MessageBubble | None = None
        self._placeholder_label: QLabel | None = None
        self._show_placeholder()

    # ---- placeholder -------------------------------------------------

    def _show_placeholder(self) -> None:
        if self._placeholder_label is not None:
            return
        self._placeholder_label = QLabel(
            "Type a request below.   e.g. 이 z-stack에서 세포 찾고 채널2 강도 측정해줘"
        )
        self._placeholder_label.setObjectName("chatPlaceholder")
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        self._vbox.insertWidget(0, self._placeholder_label)

    def _hide_placeholder(self) -> None:
        if self._placeholder_label is None:
            return
        self._placeholder_label.deleteLater()
        self._placeholder_label = None

    # ---- public API --------------------------------------------------

    def append_user(self, text: str) -> None:
        self._hide_placeholder()
        bubble = MessageBubble("user", text)
        self._add_row(bubble, align="right")
        self._current_assistant = None

    def append_system(self, text: str) -> None:
        self._hide_placeholder()
        bubble = MessageBubble("system", text)
        self._add_row(bubble, align="center")
        self._current_assistant = None

    def begin_assistant(self) -> MessageBubble:
        self._hide_placeholder()
        bubble = MessageBubble("assistant", "")
        self._add_row(bubble, align="left")
        self._current_assistant = bubble
        return bubble

    def append_assistant_delta(self, delta: str) -> None:
        if self._current_assistant is None:
            self.begin_assistant()
        assert self._current_assistant is not None
        self._current_assistant.append_text(delta)
        self._defer_scroll_to_bottom()

    def clear(self) -> None:
        for bubble in self._bubbles:
            bubble.setParent(None)
            bubble.deleteLater()
        # Also clear any row containers we created.
        while self._vbox.count() > 1:  # keep the trailing stretch
            item = self._vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._bubbles.clear()
        self._current_assistant = None
        self._placeholder_label = None
        self._show_placeholder()

    # ---- backward-compat shims (existing test) -----------------------

    def append(self, text: str) -> None:  # noqa: A003 — match QTextEdit API
        self.append_system(text)

    def toPlainText(self) -> str:
        return "\n".join(b.text() for b in self._bubbles)

    # ---- internals ---------------------------------------------------

    def _add_row(self, bubble: MessageBubble, align: str) -> None:
        row = QWidget(self._inner)
        row.setObjectName("chatRow")
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        if align == "right":
            hbox.addStretch(1)
            hbox.addWidget(bubble)
        elif align == "left":
            hbox.addWidget(bubble)
            hbox.addStretch(1)
        else:  # center / system
            hbox.addStretch(1)
            hbox.addWidget(bubble)
            hbox.addStretch(1)

        # Insert above the trailing stretch.
        insert_at = self._vbox.count() - 1
        self._vbox.insertWidget(insert_at, row)
        self._bubbles.append(bubble)
        self._defer_scroll_to_bottom()

    def _defer_scroll_to_bottom(self) -> None:
        # Defer to next event-loop iteration so layout has actually updated.
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def resizeEvent(self, event) -> None:  # cap bubble width to ratio of viewport
        super().resizeEvent(event)
        viewport_w = self.viewport().width()
        max_w = int(viewport_w * self.BUBBLE_MAX_RATIO)
        for b in self._bubbles:
            if b.role != "system":
                b.setMaximumWidth(max_w)
