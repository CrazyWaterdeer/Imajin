from __future__ import annotations

import subprocess
import sys


def test_cli_doctor_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "imajin.cli", "--doctor"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "imajin doctor" in result.stdout


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "imajin.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--doctor" in result.stdout


def test_input_method_env_follows_existing_ime(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    monkeypatch.setenv("GTK_IM_MODULE", "fcitx")

    cli._setup_input_method_env()

    assert cli.os.environ["QT_IM_MODULE"] == "fcitx"


def test_input_method_env_does_not_override_user_choice(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.setenv("QT_IM_MODULE", "ibus")
    monkeypatch.setenv("GTK_IM_MODULE", "fcitx")

    cli._setup_input_method_env()

    assert cli.os.environ["QT_IM_MODULE"] == "ibus"


def test_input_method_env_defaults_to_wayland_on_wsl(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)
    monkeypatch.delenv("XMODIFIERS", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland;xcb")
    monkeypatch.setattr(cli, "_is_wsl", lambda: True)

    cli._setup_input_method_env()

    assert cli.os.environ["QT_IM_MODULE"] == "wayland"
