from __future__ import annotations

import socket
from unittest.mock import MagicMock, mock_open, patch


from imajin.ui import ollama_helper


def test_is_running_true_when_socket_connects() -> None:
    fake = MagicMock()
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    with patch.object(ollama_helper.socket, "create_connection", return_value=fake):
        assert ollama_helper.is_running("http://localhost:11434/v1") is True


def test_is_running_false_when_socket_refused() -> None:
    with patch.object(
        ollama_helper.socket, "create_connection", side_effect=ConnectionRefusedError
    ):
        assert ollama_helper.is_running("http://localhost:11434/v1") is False


def test_is_running_false_when_timeout() -> None:
    with patch.object(
        ollama_helper.socket, "create_connection", side_effect=socket.timeout
    ):
        assert ollama_helper.is_running("http://localhost:11434/v1") is False


def test_host_port_from_default_url() -> None:
    assert ollama_helper._host_port("http://localhost:11434/v1") == ("localhost", 11434)
    assert ollama_helper._host_port("http://10.0.0.5:9000") == ("10.0.0.5", 9000)
    assert ollama_helper._host_port("http://example.com/x") == ("example.com", 11434)


def test_ensure_running_already_running() -> None:
    with patch.object(ollama_helper, "is_running", return_value=True):
        assert ollama_helper.ensure_running() == "already-running"


def test_ensure_running_not_installed() -> None:
    with (
        patch.object(ollama_helper, "is_running", return_value=False),
        patch.object(ollama_helper, "is_installed", return_value=False),
    ):
        assert ollama_helper.ensure_running() == "not-installed"


def test_ensure_running_started_when_spawn_succeeds() -> None:
    with (
        patch.object(ollama_helper, "is_running", return_value=False),
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper, "start_daemon", return_value=True),
    ):
        assert ollama_helper.ensure_running() == "started"


def test_ensure_running_start_failed_when_popen_errors() -> None:
    with (
        patch.object(ollama_helper, "is_running", return_value=False),
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper, "start_daemon", return_value=False),
    ):
        assert ollama_helper.ensure_running() == "start-failed"


def test_start_daemon_invokes_ollama_serve_detached() -> None:
    # No os.name patch: this asserts the real-platform (POSIX, in CI) branch.
    with (
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper.subprocess, "Popen") as popen,
    ):
        assert ollama_helper.start_daemon() is True
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == ["ollama", "serve"]
    assert kwargs["start_new_session"] is True


def test_start_daemon_returns_false_when_oserror() -> None:
    with (
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper.subprocess, "Popen", side_effect=OSError),
    ):
        assert ollama_helper.start_daemon() is False


def test_start_daemon_not_installed_returns_false_without_spawning() -> None:
    with (
        patch.object(ollama_helper, "is_installed", return_value=False),
        patch.object(ollama_helper.subprocess, "Popen") as popen,
    ):
        assert ollama_helper.start_daemon() is False
    popen.assert_not_called()


def test_start_daemon_windows_uses_detached_creationflags() -> None:
    # subprocess.CREATE_NO_WINDOW / DETACHED_PROCESS don't exist on this (POSIX)
    # interpreter, so create=True adds them for the duration of the test — this
    # is exactly the getattr(..., 0) fallback in start_daemon() earning its keep.
    with (
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper.os, "name", "nt"),
        patch.object(ollama_helper.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
        patch.object(ollama_helper.subprocess, "DETACHED_PROCESS", 0x00000008, create=True),
        patch.object(ollama_helper.subprocess, "Popen") as popen,
    ):
        assert ollama_helper.start_daemon() is True
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == ["ollama", "serve"]
    assert "start_new_session" not in kwargs
    assert kwargs["creationflags"] == 0x08000008
    assert kwargs["close_fds"] is True
    assert kwargs["stdout"] == ollama_helper.subprocess.DEVNULL
    assert kwargs["stderr"] == ollama_helper.subprocess.DEVNULL
    assert kwargs["stdin"] == ollama_helper.subprocess.DEVNULL


def test_start_daemon_windows_returns_false_when_oserror() -> None:
    with (
        patch.object(ollama_helper, "is_installed", return_value=True),
        patch.object(ollama_helper.os, "name", "nt"),
        patch.object(ollama_helper.subprocess, "Popen", side_effect=OSError),
    ):
        assert ollama_helper.start_daemon() is False


def test_start_daemon_windows_not_installed_returns_false_without_spawning() -> None:
    with (
        patch.object(ollama_helper, "is_installed", return_value=False),
        patch.object(ollama_helper.os, "name", "nt"),
        patch.object(ollama_helper.subprocess, "Popen") as popen,
    ):
        assert ollama_helper.start_daemon() is False
    popen.assert_not_called()


def test_wsl_host_ip_reads_nameserver_from_resolv_conf() -> None:
    resolv_conf = "# auto-generated by WSL\nnameserver 172.29.16.1\n"
    with (
        patch.object(ollama_helper, "_is_wsl", return_value=True),
        patch.object(ollama_helper, "open", mock_open(read_data=resolv_conf), create=True),
    ):
        assert ollama_helper.wsl_host_ip() == "172.29.16.1"


def test_wsl_host_ip_none_when_not_wsl() -> None:
    # Fake resolv.conf too, so a bug that skips the WSL check can't pass by
    # accident just because this sandbox happens to run on real WSL.
    with (
        patch.object(ollama_helper, "_is_wsl", return_value=False),
        patch.object(
            ollama_helper, "open", mock_open(read_data="nameserver 172.29.16.1\n"), create=True
        ),
    ):
        assert ollama_helper.wsl_host_ip() is None


def test_wsl_host_ip_none_when_resolv_conf_unreadable() -> None:
    with (
        patch.object(ollama_helper, "_is_wsl", return_value=True),
        patch.object(ollama_helper, "open", side_effect=OSError, create=True),
    ):
        assert ollama_helper.wsl_host_ip() is None


def test_wsl_host_ip_none_when_no_nameserver_line() -> None:
    with (
        patch.object(ollama_helper, "_is_wsl", return_value=True),
        patch.object(ollama_helper, "open", mock_open(read_data="# empty\n"), create=True),
    ):
        assert ollama_helper.wsl_host_ip() is None


def test_suggest_base_url_substitutes_host_on_wsl() -> None:
    with patch.object(ollama_helper, "wsl_host_ip", return_value="172.29.16.1"):
        assert (
            ollama_helper.suggest_base_url("http://localhost:11434/v1")
            == "http://172.29.16.1:11434/v1"
        )


def test_suggest_base_url_none_when_not_wsl() -> None:
    with patch.object(ollama_helper, "wsl_host_ip", return_value=None):
        assert ollama_helper.suggest_base_url("http://localhost:11434/v1") is None


def test_suggest_base_url_none_when_host_is_not_localhost() -> None:
    with patch.object(ollama_helper, "wsl_host_ip", return_value="172.29.16.1"):
        assert ollama_helper.suggest_base_url("http://10.0.0.5:11434/v1") is None
