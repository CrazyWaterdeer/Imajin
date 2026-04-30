from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs


@dataclass
class Settings:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"
    data_dir: Path = field(
        default_factory=lambda: Path(platformdirs.user_data_dir("imajin"))
    )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
        )

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def templates_dir(self) -> Path:
        return self.data_dir / "templates"


def ensure_dirs(settings: Settings) -> None:
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    settings.templates_dir.mkdir(parents=True, exist_ok=True)
