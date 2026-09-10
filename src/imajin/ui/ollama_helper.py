"""Detect / auto-start Ollama daemon for local model use.

If `ollama` is installed but no daemon is running, we spawn one detached so
imajin closing won't kill it — on POSIX via start_new_session, on Windows via
CREATE_NO_WINDOW | DETACHED_PROCESS (see start_daemon). If Ollama isn't
installed (e.g. laptop), all calls are no-ops and the chat dock will mark
Ollama models as offline.

is_running() only proves the port is open, not that the daemon is actually
usable (see its docstring). And on WSL, Ollama commonly runs on the Windows
host rather than inside the Linux VM, so 'localhost' is refused across the
NAT — see wsl_host_ip() / suggest_base_url().
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
    """Cheap TCP liveness check: is anything listening on the Ollama port?

    This proves only that the socket accepts a connection — a daemon that is up
    but has zero models pulled, or none that support tool calls, answers True
    here too, and the chat dock's first real request then dies with a raw 404.
    Callers that need to know the daemon is actually *usable* should pair this
    with :func:`imajin.agent.local_models.probe_ollama` (GET /api/tags + POST
    /api/show), which this module deliberately does not import — it needs to
    stay import-light since it runs during UI startup, so the caller composes
    the two rather than this module reaching into agent/.
    """
    host, port = _host_port(base_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_daemon() -> bool:
    """Spawn `ollama serve` detached so it outlives imajin. Returns True if the
    spawn succeeded — not that the daemon is up yet; poll is_running() for that.

    Windows: CREATE_NO_WINDOW suppresses the console flash a plain Popen would
    otherwise pop up, and DETACHED_PROCESS takes the child out of imajin's
    console session so it isn't torn down with it. Both are looked up via
    getattr(..., 0) so this still runs (as a harmless no-op flag) on POSIX,
    where `subprocess` defines neither attribute — the branch itself is dead
    there since os.name != "nt", but the module must still *import* cleanly.
    close_fds=True is safe together with the DEVNULL redirects here: since
    Python 3.7, Windows CreateProcess selectively inherits only the redirected
    handles via STARTUPINFO's handle list, rather than requiring
    close_fds=False to inherit any handle at all (verified against this
    interpreter's subprocess.py — _execute_child builds `handle_list` from
    p2cread/c2pwrite/errwrite whenever close_fds and use_std_handles are both
    true, it does not raise).

    POSIX: start_new_session=True puts the daemon in its own session so it is
    outside imajin's process group and isn't signalled when imajin's shell
    (or terminal) exits.

    Returns False if `ollama` isn't installed, or if the spawn itself raises
    OSError (e.g. the binary vanished between is_installed() and here).
    """
    if not is_installed():
        return False
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "DETACHED_PROCESS", 0)
                ),
                close_fds=True,
            )
        else:
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


def _is_wsl() -> bool:
    """Mirrors imajin.cli._is_wsl() exactly (duplicated, not imported — cli.py
    pulls in napari/Qt and other heavy imports this module must stay free of)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def wsl_host_ip() -> str | None:
    """The Windows host's IP as seen from inside WSL2, or None off-WSL/unreadable.

    WSL2 is a lightweight VM behind a NAT, so 'localhost' inside it names the
    VM, not the Windows host — an Ollama daemon serving on the Windows side is
    unreachable there. /etc/resolv.conf's nameserver line is written by WSL2 to
    the NAT gateway's IP, which doubles as the address the Windows host is
    reachable at from inside the VM; this is the standard WSL2
    reach-the-host trick (the same one used for X11/display forwarding).
    """
    if not _is_wsl():
        return None
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    return parts[1]
    except OSError:
        return None
    return None


def suggest_base_url(current: str) -> str | None:
    """A host-IP-substituted URL to suggest when `current` (localhost) is refused.

    Only produces a suggestion when `current` actually points at localhost on
    WSL — if the user already configured a real host/IP, or we're not on WSL,
    there's nothing useful to swap in and we return None.
    """
    parsed = urlparse(current)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return None
    host_ip = wsl_host_ip()
    if not host_ip:
        return None
    netloc = f"{host_ip}:{parsed.port}" if parsed.port else host_ip
    return parsed._replace(netloc=netloc).geturl()


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
