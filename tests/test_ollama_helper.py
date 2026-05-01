from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

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
