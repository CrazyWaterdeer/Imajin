from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from imajin.paths import is_wsl, normalize_user_path, windows_drive_roots


KST = timezone(timedelta(hours=9), name="KST")


def _kst_now() -> datetime:
    """Return current time in KST (UTC+9), used for bundle folder timestamps."""
    return datetime.now(KST)


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
        from imajin.project import current_project

        project = current_project()
    except Exception:
        project = None
    if project is not None:
        return project.path / "reports"
    return user_results_root()


def results_dir(category: str) -> Path:
    return results_root() / category


def slugify_result_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "result"


def unique_result_path(category: str, filename: str) -> Path:
    root = results_dir(category)
    path = root / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        candidate = root / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def unique_result_dir(category: str, dirname: str) -> Path:
    root = results_dir(category)
    base = slugify_result_name(dirname)
    path = root / base
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = root / f"{base}_{i}"
        if not candidate.exists():
            return candidate
        i += 1


def create_result_bundle(
    name: str,
    *,
    kind: str = "analysis",
    metadata: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    bundle = unique_result_dir("bundles", f"{timestamp}_{slugify_result_name(name)}")
    for subdir in ("labels", "tables", "qc", "figures"):
        (bundle / subdir).mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
        "outputs": {},
    }
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


def record_result(kind: str, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    root = results_root()
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": kind,
        "path": str(path),
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
    }
    manifest = root / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
