from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from imajin.paths import is_wsl, normalize_user_path, windows_drive_roots


_KST = timezone(timedelta(hours=9), name="KST")


def _kst_now() -> datetime:
    """Return current time in KST (UTC+9), used for bundle folder timestamps."""
    return datetime.now(_KST)


def _git_commit_short() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _collect_env_info() -> dict[str, Any]:
    """Collect package and platform versions for embedding in bundle metadata."""
    import platform
    from importlib import metadata as _metadata

    def _ver(pkg: str) -> str | None:
        try:
            return _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            return None

    deps_to_record = (
        "cellpose",
        "scikit-image",
        "tifffile",
        "numpy",
        "pandas",
        "napari",
    )
    deps = {pkg: _ver(pkg) for pkg in deps_to_record}
    deps = {k: v for k, v in deps.items() if v is not None}

    return {
        "python_version": platform.python_version(),
        "imajin_version": _ver("imajin"),
        "deps": deps,
        "git_commit": _git_commit_short(),
    }


def _windows_documents_dir() -> Path | None:
    if not is_wsl():
        return None
    home_name = Path.home().name.lower()
    for root in windows_drive_roots():
        users = root / "Users"
        if not users.is_dir():
            continue
        candidates = sorted(
            [p for p in users.iterdir() if p.is_dir()],
            key=lambda p: (p.name.lower() != home_name, p.name.lower()),
        )
        for user_dir in candidates:
            if user_dir.name.lower() in {"public", "default"}:
                continue
            documents = user_dir / "Documents"
            if documents.is_dir():
                return documents
    return None


def user_results_root() -> Path:
    configured = os.environ.get("IMAJIN_RESULTS_DIR")
    if configured:
        return normalize_user_path(configured).expanduser()

    windows_documents = _windows_documents_dir()
    if windows_documents is not None:
        return windows_documents / "Imajin" / "results"

    documents = Path.home() / "Documents"
    if documents.is_dir():
        return documents / "Imajin" / "results"
    return Path.home() / "Imajin" / "results"


def results_root() -> Path:
    try:
        from imajin.anchor import resolve_session_anchor

        anchor = resolve_session_anchor()
    except Exception:
        anchor = None
    if anchor is not None:
        return anchor

    return user_results_root()


def slugify_result_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "result"


def _unique_subdir(root: Path, dirname: str) -> Path:
    """Create a unique <root>/<slugified-dirname>[_N] path without colliding."""
    root.mkdir(parents=True, exist_ok=True)
    base = slugify_result_name(dirname)
    candidate = root / base
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = root / f"{base}_{i}"
        if not candidate.exists():
            return candidate
        i += 1


def create_result_bundle(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: Path | str | None = None,
) -> Path:
    """Create a fresh bundle directory with the standard layout and seed metadata.

    Layout:
        <ts>_<name>/
        ├── metadata.json

    Output subdirectories are created lazily when a writer actually emits files.

    `kind` is "single" or "batch"; `tier` is "single_tier" or "two_tier" (or
    None when not yet decided — the caller can update via write_bundle_metadata).
    Extra metadata is merged at the top level of metadata.json.
    """
    now = _kst_now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    if root is None:
        root = user_results_root()
    bundle = _unique_subdir(Path(root), f"{timestamp}_{slugify_result_name(name)}")
    env = _collect_env_info()
    payload: dict[str, Any] = dict(metadata or {})
    payload.update({
        "kind": kind,
        "tier": tier,
        "name": name,
        "status": "in_progress",
        "created_at": now.isoformat(),
        **env,
    })
    write_bundle_metadata(bundle, payload)
    return bundle


def write_bundle_metadata(bundle: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(bundle)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def read_bundle_metadata(bundle: str | Path) -> dict[str, Any]:
    path = Path(bundle) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


