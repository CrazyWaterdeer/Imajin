from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import platformdirs


@dataclass
class Settings:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"
    ui_scale: str = "auto"
    data_dir: Path = field(
        default_factory=lambda: Path(platformdirs.user_data_dir("imajin"))
    )

    PERSIST_FIELDS: ClassVar[tuple[str, ...]] = (
        "anthropic_api_key",
        "openai_api_key",
        "openai_base_url",
        "ollama_base_url",
        "ui_scale",
    )

    @classmethod
    def secrets_path(cls) -> Path:
        return Path(platformdirs.user_config_dir("imajin")) / "secrets.json"

    @classmethod
    def _read_secrets_file(cls) -> dict:
        path = cls.secrets_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    @classmethod
    def from_env(cls) -> "Settings":
        file_data = cls._read_secrets_file()

        def pick(env_key: str, file_key: str, default: str | None = None) -> str | None:
            return os.environ.get(env_key) or file_data.get(file_key) or default

        return cls(
            anthropic_api_key=pick("ANTHROPIC_API_KEY", "anthropic_api_key"),
            openai_api_key=pick("OPENAI_API_KEY", "openai_api_key"),
            openai_base_url=pick(
                "OPENAI_BASE_URL", "openai_base_url", "https://api.openai.com/v1"
            ),
            ollama_base_url=pick(
                "OLLAMA_BASE_URL", "ollama_base_url", "http://localhost:11434/v1"
            ),
            ui_scale=file_data.get("ui_scale") or "auto",
        )

    def save_secrets(self) -> None:
        path = self.secrets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f: getattr(self, f) for f in self.PERSIST_FIELDS}
        path.write_text(json.dumps(data, indent=2))
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def templates_dir(self) -> Path:
        return self.data_dir / "templates"


def ensure_dirs(settings: Settings) -> None:
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    settings.templates_dir.mkdir(parents=True, exist_ok=True)
