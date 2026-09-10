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


def test_model_choice_persists_in_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings()
        s.default_provider = "claude-agent"
        s.default_model = "opus"
        s.save_secrets()

        raw = json.loads(secrets.read_text())
        assert raw["default_provider"] == "claude-agent"
        assert raw["default_model"] == "opus"


def test_from_env_reads_model_choice_from_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"default_provider": "openai", "default_model": "gpt"}))
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert (s.default_provider, s.default_model) == ("openai", "gpt")


def test_from_env_defaults_model_choice_when_missing(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"anthropic_api_key": "sk-x"}))
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert (s.default_provider, s.default_model) == ("anthropic", "sonnet")


def test_ollama_model_defaults_to_blank() -> None:
    # Blank means "nothing configured" -- the model picker then shows an
    # unconfirmed placeholder row rather than a fabricated model name.
    s = Settings()
    assert s.ollama_model == ""


def test_ollama_model_persists_in_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings()
        s.ollama_model = "qwen3.5:9b"
        s.save_secrets()

        raw = json.loads(secrets.read_text())
        assert raw["ollama_model"] == "qwen3.5:9b"


def test_from_env_reads_ollama_model_from_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"ollama_model": "llama3.1:8b"}))
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert s.ollama_model == "llama3.1:8b"


def test_from_env_prefers_ollama_model_env_var_over_file(
    tmp_path: Path, monkeypatch
) -> None:
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"ollama_model": "from-file:1b"}))
    monkeypatch.setenv("OLLAMA_MODEL", "from-env:1b")
    with patch.object(Settings, "secrets_path", classmethod(lambda cls: secrets)):
        s = Settings.from_env()
        assert s.ollama_model == "from-env:1b"
