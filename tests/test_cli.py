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


def test_input_method_env_forces_xcb_on_wsl_when_ime_present(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    monkeypatch.setenv("GTK_IM_MODULE", "fcitx")
    monkeypatch.setenv("XMODIFIERS", "@im=fcitx")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland;xcb")
    monkeypatch.setattr(cli, "_is_wsl", lambda: True)

    cli._setup_input_method_env()

    assert cli.os.environ["QT_IM_MODULE"] == "fcitx"
    # fcitx's XIM bridge only reaches Qt under XWayland, not WSLg's Wayland.
    assert cli.os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_detect_desktop_ime_finds_installed_fcitx_binary(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)
    monkeypatch.delenv("XMODIFIERS", raising=False)
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/usr/bin/fcitx5" if name == "fcitx5" else None
    )

    assert cli._detect_desktop_ime() == "fcitx"


def test_input_method_env_no_engine_leaves_qt_im_module_unset(monkeypatch) -> None:
    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)
    monkeypatch.delenv("XMODIFIERS", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "_is_wsl", lambda: True)

    cli._setup_input_method_env()

    # No IME engine → do NOT set the wayland stub that swallowed keystrokes.
    assert "QT_IM_MODULE" not in cli.os.environ


def test_ensure_fcitx_starts_in_xim_mode_without_wayland(monkeypatch) -> None:
    import subprocess as sp
    from types import SimpleNamespace

    from imajin import cli

    monkeypatch.setenv("QT_IM_MODULE", "fcitx")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(cli, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/fcitx5")
    monkeypatch.setattr(sp, "run", lambda *a, **k: SimpleNamespace(returncode=1))  # not running

    captured: dict = {}
    monkeypatch.setattr(sp, "Popen", lambda args, **k: captured.update(args=args, env=k.get("env")))

    cli._ensure_fcitx()

    assert captured["args"][:2] == ["fcitx5", "-d"]
    # WSLg denies the Wayland IME protocol — fcitx5 must run X11/XIM-only.
    assert "WAYLAND_DISPLAY" not in captured["env"]


def test_ensure_fcitx_noop_when_fcitx_not_selected(monkeypatch) -> None:
    import subprocess as sp

    from imajin import cli

    monkeypatch.delenv("QT_IM_MODULE", raising=False)
    called: list = []
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: called.append(1))

    cli._ensure_fcitx()

    assert called == []
