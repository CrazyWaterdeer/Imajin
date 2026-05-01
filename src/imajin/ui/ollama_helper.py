"""Detect / auto-start Ollama daemon for local model use.

If `ollama` is installed but no daemon is running, we spawn one detached so
imajin closing won't kill it. If Ollama isn't installed (e.g. laptop), all
calls are no-ops and the chat dock will mark Ollama models as offline.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from urllib.parse import urlparse


def is_installed() -> bool:
    return shutil.which("ollama") is not None


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    return host, port


def is_running(base_url: str = "http://localhost:11434/v1", timeout: float = 0.5) -> bool:
    host, port = _host_port(base_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_daemon() -> bool:
    """Spawn `ollama serve` detached. Returns True if the spawn succeeded.

    POSIX-only: relies on `start_new_session=True` so the daemon survives
    imajin exit. Returns False on non-POSIX or any spawn failure.
    """
    if os.name != "posix":
        return False
    if not is_installed():
        return False
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except OSError:
        return False


def ensure_running(base_url: str = "http://localhost:11434/v1") -> str:
    """Make Ollama available if possible. Returns one of:

    - "already-running" : daemon was already up.
    - "started"         : we spawned the daemon (still warming up — caller
                          should not assume it's immediately reachable).
    - "not-installed"   : ollama binary missing (laptop case); no-op.
    - "start-failed"    : tried to spawn but Popen failed.
    """
    if is_running(base_url):
        return "already-running"
    if not is_installed():
        return "not-installed"
    return "started" if start_daemon() else "start-failed"
