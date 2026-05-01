from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from imajin.config import Settings
from imajin.ui.theme import Theme, apply_dock_theme


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Imajin — API Keys")
        apply_dock_theme(self)
        self.setMinimumSize(520, 240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "Stored at <code>{}</code>.<br>"
            "Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "OLLAMA_BASE_URL) take precedence at startup."
            .format(Settings.secrets_path())
        )
        intro.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-weight: normal;")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(form.labelAlignment())
        form.setSpacing(10)

        self.anthropic_edit = QLineEdit(settings.anthropic_api_key or "")
        self.anthropic_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_edit.setPlaceholderText("sk-ant-...")
        form.addRow("Anthropic API Key", self.anthropic_edit)

        self.openai_edit = QLineEdit(settings.openai_api_key or "")
        self.openai_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_edit.setPlaceholderText("sk-...")
        form.addRow("OpenAI API Key", self.openai_edit)

        self.ollama_edit = QLineEdit(settings.ollama_base_url)
        self.ollama_edit.setPlaceholderText("http://localhost:11434/v1")
        form.addRow("Ollama base URL", self.ollama_edit)

        layout.addLayout(form)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setObjectName("sendBtn")
        save_btn.setStyleSheet(save_btn.styleSheet())
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        self.settings.anthropic_api_key = self.anthropic_edit.text().strip() or None
        self.settings.openai_api_key = self.openai_edit.text().strip() or None
        self.settings.ollama_base_url = (
            self.ollama_edit.text().strip() or "http://localhost:11434/v1"
        )
        self.settings.save_secrets()
        self.accept()
