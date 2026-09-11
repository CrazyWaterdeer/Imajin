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


def test_doctor_providers_section_reports_each_kind(monkeypatch, capsys) -> None:
    # In-process (not subprocess) so compute_statuses/discover_ollama_models can
    # be faked -- house rule is no real network/Ollama in tests, and a subprocess
    # run can't be monkeypatched from here.
    from imajin import cli
    from imajin.config import Settings
    from imajin.ui.provider_status import ProviderStatus

    monkeypatch.setattr(
        "imajin.ui.provider_status.compute_statuses",
        lambda settings: {
            "anthropic": ProviderStatus(True, None),
            "claude-agent": ProviderStatus(False, "not logged in"),
            "openai": ProviderStatus(False, "no API key"),
            "ollama": ProviderStatus(False, "no tool-capable model"),
        },
    )
    monkeypatch.setattr(
        "imajin.agent.local_models.discover_ollama_models", lambda base_url, **kw: []
    )

    cli._doctor(Settings())

    out = capsys.readouterr().out
    assert "[Providers]" in out
    assert "anthropic" in out
    assert "not logged in" in out
    assert "no tool-capable model" in out
    assert "[Ollama]" in out
    assert "no models discovered" in out


def test_doctor_ollama_section_lists_discovered_models(monkeypatch, capsys) -> None:
    from imajin import cli
    from imajin.agent.local_models import LocalModel
    from imajin.config import Settings
    from imajin.ui.provider_status import ProviderStatus

    monkeypatch.setattr(
        "imajin.ui.provider_status.compute_statuses",
        lambda settings: {
            k: ProviderStatus(True, None)
            for k in ("anthropic", "claude-agent", "openai", "ollama")
        },
    )
    model = LocalModel(
        name="qwen3.5:9b",
        context_length=262144,
        capabilities=frozenset({"completion", "tools", "vision", "thinking"}),
        parameter_size="9.7B",
        size_bytes=6_000_000_000,
    )
    monkeypatch.setattr(
        "imajin.agent.local_models.discover_ollama_models", lambda base_url, **kw: [model]
    )

    cli._doctor(Settings())

    out = capsys.readouterr().out
    assert "qwen3.5:9b" in out
    assert "9.7B" in out
    assert "262144" in out
    assert "tools" in out


def test_doctor_return_code_unaffected_by_provider_availability(monkeypatch) -> None:
    # Providers are informational -- the exit code contract (0 ok / 1 not) stays
    # reserved for imports/CUDA/display, so all-down vs all-up must not change
    # it. Comparing two runs rather than asserting a literal 0/1 keeps this
    # independent of whatever imports/CUDA/display looks like on the machine
    # running the test.
    from imajin import cli
    from imajin.config import Settings
    from imajin.ui.provider_status import ProviderStatus

    monkeypatch.setattr(
        "imajin.agent.local_models.discover_ollama_models", lambda base_url, **kw: []
    )

    monkeypatch.setattr(
        "imajin.ui.provider_status.compute_statuses",
        lambda settings: {
            k: ProviderStatus(False, "unavailable")
            for k in ("anthropic", "claude-agent", "openai", "ollama")
        },
    )
    code_all_down = cli._doctor(Settings())

    monkeypatch.setattr(
        "imajin.ui.provider_status.compute_statuses",
        lambda settings: {
            k: ProviderStatus(True, None)
            for k in ("anthropic", "claude-agent", "openai", "ollama")
        },
    )
    code_all_up = cli._doctor(Settings())

    assert code_all_down == code_all_up


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
