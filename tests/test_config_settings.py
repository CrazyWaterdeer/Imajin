from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from imajin.config import Settings


def test_ui_scale_default_is_auto() -> None:
    s = Settings()
    assert s.ui_scale == "auto"


def test_ui_scale_persists_in_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings()
        s.ui_scale = "1.5"
        s.anthropic_api_key = "sk-test"
        s.save_secrets()

        raw = json.loads(secrets.read_text())
        assert raw["ui_scale"] == "1.5"
        assert raw["anthropic_api_key"] == "sk-test"


def test_from_env_reads_ui_scale_from_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"ui_scale": "1.25"}))
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert s.ui_scale == "1.25"


def test_from_env_defaults_ui_scale_to_auto_when_missing(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"anthropic_api_key": "sk-x"}))
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert s.ui_scale == "auto"
