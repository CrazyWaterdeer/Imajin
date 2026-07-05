from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")
_WSL_UNC_RE = re.compile(
    r"^[\\/]{2}wsl(?:\.localhost|\$)?[\\/]+([^\\/]+)(?:[\\/]+(.*))?$",
    re.IGNORECASE,
)

# Entries under Windows ``C:\Users`` that are junctions or system profiles, not
# real people. Excluded when picking a default user profile so we never land on a
# reparse point like ``All Users`` (→ ProgramData) or ``Default User`` (→ Default).
_NON_USER_PROFILES = frozenset(
    {"public", "default", "default user", "all users", "defaultuser0"}
)


def is_wsl() -> bool:
    """Return True when the process appears to be running under WSL."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    version = version.lower()
    return "microsoft" in version or "wsl" in version


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    wrappers = (('"', '"'), ("'", "'"), ("`", "`"), ("<", ">"))
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
                break
    return text


def _file_uri_to_path(text: str) -> str:
    if not text.lower().startswith("file:"):
        return text
    parsed = urlparse(text)
    local = unquote(parsed.path or "")
    if parsed.netloc.lower() in {"wsl.localhost", "wsl$"}:
        return f"//{parsed.netloc}{local}"
    if re.match(r"^/[A-Za-z]:[\\/]", local):
        return local[1:]
    if parsed.netloc and _WINDOWS_DRIVE_RE.match(parsed.netloc):
        return f"{parsed.netloc}{local}"
    return local or text


def _wsl_unc_to_path(text: str) -> Path | None:
    match = _WSL_UNC_RE.match(text)
    if not match or os.name == "nt":
        return None
    rest = (match.group(2) or "").replace("\\", "/").strip("/")
    return Path("/") / rest if rest else Path("/")


def normalize_user_path(path: str | os.PathLike[str]) -> Path:
    """Resolve user-provided paths from either Linux/WSL or Windows syntax."""
    text = _file_uri_to_path(_strip_wrapping(os.fspath(path)))
    wsl_path = _wsl_unc_to_path(text)
    if wsl_path is not None:
        return wsl_path
    match = _WINDOWS_DRIVE_RE.match(text)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path("/mnt") / drive / rest
    return Path(text).expanduser()


def _safe_is_dir(path: Path) -> bool:
    """``Path.is_dir`` that treats an unreadable/dead mount as "not a directory".

    ``Path.is_dir()`` swallows a handful of errnos (ENOENT/ENOTDIR/...) but not
    ENODEV/ENXIO, so a stale WSL drive mount — e.g. an ejected USB that was ``E:``
    or a disconnected network drive — makes the bare call raise
    ``OSError: [Errno 19] No such device``. Enumerating drives or dialog
    locations must skip such an entry, not crash the caller.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def windows_drive_roots() -> list[Path]:
    roots: list[Path] = []
    mnt = Path("/mnt")
    if not mnt.exists():
        return roots
    try:
        children = sorted(mnt.iterdir())
    except OSError:
        return roots
    for child in children:
        if len(child.name) == 1 and child.name.isalpha() and _safe_is_dir(child):
            roots.append(child)
    return roots


def windows_file_dialog_locations() -> list[Path]:
    """Useful sidebar/start locations when the app runs under WSL."""
    locations: list[Path] = []
    for root in windows_drive_roots():
        locations.append(root)
        users = root / "Users"
        if not _safe_is_dir(users):
            continue
        try:
            user_dirs = sorted(users.iterdir())
        except OSError:
            continue
        for user_dir in user_dirs:
            if not _safe_is_dir(user_dir) or user_dir.name.lower() in _NON_USER_PROFILES:
                continue
            locations.append(user_dir)
            downloads = user_dir / "Downloads"
            desktop = user_dir / "Desktop"
            documents = user_dir / "Documents"
            for candidate in (downloads, desktop, documents):
                if _safe_is_dir(candidate):
                    locations.append(candidate)
    deduped: list[Path] = []
    seen: set[str] = set()
    for loc in locations:
        key = str(loc)
        if key not in seen:
            deduped.append(loc)
            seen.add(key)
    return deduped
