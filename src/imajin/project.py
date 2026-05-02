from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from imajin import __version__

CURRENT_SCHEMA_VERSION = 1


@dataclass
class ProjectMetadata:
    schema_version: int
    project_id: str
    name: str
    created_at: str
    updated_at: str
    imajin_version: str
    notes: str = ""


@dataclass
class ProjectContext:
    path: Path
    metadata: ProjectMetadata


_CURRENT_PROJECT: ProjectContext | None = None
_AUTOSAVE_DEPTH = 0
_LAST_AUTOSAVE: dict[str, Any] | None = None
_LAST_AUTOSAVE_ERROR: str | None = None


def current_project() -> ProjectContext | None:
    return _CURRENT_PROJECT


def close_project() -> None:
    global _CURRENT_PROJECT, _LAST_AUTOSAVE, _LAST_AUTOSAVE_ERROR
    _CURRENT_PROJECT = None
    _LAST_AUTOSAVE = None
    _LAST_AUTOSAVE_ERROR = None


def create_project(path: str | Path, name: str | None = None, notes: str = "") -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _ensure_layout(root)

    now = _now()
    metadata = ProjectMetadata(
        schema_version=CURRENT_SCHEMA_VERSION,
        project_id=f"proj_{datetime.now(UTC).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
        name=name or root.stem,
        created_at=now,
        updated_at=now,
        imajin_version=__version__,
        notes=notes,
    )

    global _CURRENT_PROJECT
    _CURRENT_PROJECT = ProjectContext(path=root, metadata=metadata)
    _write_initial_files(root, metadata)
    return _project_result(root, metadata, created=True)


def save_project(path: str | Path | None = None) -> dict[str, Any]:
    global _CURRENT_PROJECT, _LAST_AUTOSAVE, _LAST_AUTOSAVE_ERROR

    if path is not None:
        root = Path(path).expanduser().resolve()
        if _CURRENT_PROJECT is None or _CURRENT_PROJECT.path != root:
            create_project(root, name=root.stem)
    if _CURRENT_PROJECT is None:
        raise RuntimeError("No current project. Call create_project(path) first.")

    root = _CURRENT_PROJECT.path
    metadata = _CURRENT_PROJECT.metadata
    _ensure_layout(root)
    metadata.updated_at = _now()

    files = _file_records_for_save(root)
    state_payload = _session_state_for_save()
    table_records = _save_tables(root, state_payload["tables"])
    jobs = _jobs_for_save()
    provenance_records = _copy_current_provenance(root)

    _write_json_atomic(root / "project.json", asdict(metadata))
    _write_json_atomic(root / "files.json", files)
    _write_json_atomic(root / "samples.json", state_payload["samples"])
    _write_json_atomic(root / "channels.json", state_payload["channels"])
    _write_json_atomic(root / "recipes.json", state_payload["recipes"])
    _write_json_atomic(root / "runs.json", state_payload["runs"])
    _write_json_atomic(root / "qc.json", state_payload["qc_records"])
    _write_json_atomic(root / "jobs.json", jobs)
    _write_json_atomic(root / "tables" / "index.json", table_records)

    result = {
        **_project_result(root, metadata, created=False),
        "n_files": len(files),
        "n_samples": len(state_payload["samples"]),
        "n_channels": len(state_payload["channels"]),
        "n_recipes": len(state_payload["recipes"]),
        "n_runs": len(state_payload["runs"]),
        "n_qc_records": len(state_payload["qc_records"]),
        "n_tables": len(table_records),
        "n_jobs": len(jobs),
        "n_provenance_files": len(provenance_records),
    }
    _LAST_AUTOSAVE = result
    _LAST_AUTOSAVE_ERROR = None
    return result


def load_project(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    metadata_raw = _read_json(root / "project.json")
    metadata_raw = migrate_project(metadata_raw)
    metadata = ProjectMetadata(**metadata_raw)

    files_raw = _read_json(root / "files.json", default=[])
    files, warnings = _file_records_for_load(root, files_raw)
    samples = _read_json(root / "samples.json", default=[])
    channels = _read_json(root / "channels.json", default=[])
    recipes = _read_json(root / "recipes.json", default=[])
    runs = _read_json(root / "runs.json", default=[])
    qc_records = _read_json(root / "qc.json", default=[])

    global _CURRENT_PROJECT
    with suspend_autosave():
        from imajin.agent import state

        _CURRENT_PROJECT = ProjectContext(path=root, metadata=metadata)
        state.restore_session_state(
            files=files,
            samples=samples,
            channels=channels,
            recipes=recipes,
            runs=runs,
            qc_records=qc_records,
            clear_existing=True,
        )
        n_tables = _load_tables(root)
        n_jobs = _load_jobs(root)
    return {
        **_project_result(root, metadata, created=False),
        "n_files": len(files),
        "n_samples": len(samples),
        "n_channels": len(channels),
        "n_recipes": len(recipes),
        "n_runs": len(runs),
        "n_qc_records": len(qc_records),
        "n_tables": n_tables,
        "n_jobs": n_jobs,
        "warnings": warnings,
    }


def relink_file(file_id: str, new_path: str | Path) -> dict[str, Any]:
    from imajin.agent import state

    project = _require_project()
    p = Path(new_path).expanduser().resolve()
    if file_id not in state._FILES:
        raise KeyError(f"File id {file_id!r} not found. Available: {list(state._FILES)}")
    rec = state._FILES[file_id]
    rec.path = str(p)
    rec.original_name = p.name
    rec.load_status = "unloaded" if p.exists() else "missing"
    save_project(project.path)
    return {
        "file_id": file_id,
        "path": str(p),
        "exists": p.exists(),
        "load_status": rec.load_status,
    }


def project_status() -> dict[str, Any]:
    project = current_project()
    if project is None:
        return {
            "open": False,
            "message": "No project is open.",
        }

    from imajin.agent import state

    payload = state.snapshot_session_state()
    missing: list[dict[str, Any]] = []
    for rec in payload["files"]:
        path = Path(str(rec.get("path") or ""))
        if not path.exists():
            missing.append(
                {
                    "file_id": rec.get("file_id"),
                    "path": str(path),
                }
            )

    return {
        "open": True,
        "project_id": project.metadata.project_id,
        "name": project.metadata.name,
        "path": str(project.path),
        "schema_version": project.metadata.schema_version,
        "updated_at": project.metadata.updated_at,
        "n_files": len(payload["files"]),
        "n_samples": len(payload["samples"]),
        "n_channels": len(payload["channels"]),
        "n_recipes": len(payload["recipes"]),
        "n_runs": len(payload["runs"]),
        "n_qc_records": len(payload["qc_records"]),
        "n_tables": len(payload["tables"]),
        "n_missing_files": len(missing),
        "missing_files": missing,
        "last_autosave": _LAST_AUTOSAVE,
        "last_autosave_error": _LAST_AUTOSAVE_ERROR,
    }


def autosave_current_project(reason: str | None = None) -> dict[str, Any] | None:
    global _LAST_AUTOSAVE, _LAST_AUTOSAVE_ERROR
    if _CURRENT_PROJECT is None or _AUTOSAVE_DEPTH > 0:
        return None
    try:
        result = save_project()
    except Exception as exc:  # noqa: BLE001
        _LAST_AUTOSAVE_ERROR = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "reason": reason,
            "error": _LAST_AUTOSAVE_ERROR,
        }
    result = {**result, "ok": True, "reason": reason}
    _LAST_AUTOSAVE = result
    return result


@contextmanager
def suspend_autosave():
    global _AUTOSAVE_DEPTH
    _AUTOSAVE_DEPTH += 1
    try:
        yield SimpleNamespace(active=True)
    finally:
        _AUTOSAVE_DEPTH = max(0, _AUTOSAVE_DEPTH - 1)


def export_project_summary(path: str | Path) -> dict[str, Any]:
    project = _require_project()
    from imajin.agent import state

    payload = state.snapshot_session_state()
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {project.metadata.name}",
        "",
        f"- Project ID: {project.metadata.project_id}",
        f"- Project path: {project.path}",
        f"- Files: {len(payload['files'])}",
        f"- Samples: {len(payload['samples'])}",
        f"- Channels: {len(payload['channels'])}",
        f"- Recipes: {len(payload['recipes'])}",
        f"- Runs: {len(payload['runs'])}",
        f"- QC records: {len(payload['qc_records'])}",
        f"- Tables: {len(payload['tables'])}",
        "",
    ]
    if payload["samples"]:
        lines.append("## Samples")
        for sample in payload["samples"]:
            group = sample.get("group") or "ungrouped"
            lines.append(f"- {sample.get('sample_name')} ({group})")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out), "format": "markdown"}


def migrate_project(data: dict[str, Any]) -> dict[str, Any]:
    version = int(data.get("schema_version", 0))
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Project schema version {version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}."
        )
    if version < 1:
        raise ValueError("Unsupported project schema version; expected version 1.")
    return dict(data)


def _ensure_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("tables", "provenance", "reports", "logs"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def _write_initial_files(root: Path, metadata: ProjectMetadata) -> None:
    _write_json_atomic(root / "project.json", asdict(metadata))
    for name in (
        "files.json",
        "samples.json",
        "channels.json",
        "recipes.json",
        "runs.json",
        "qc.json",
        "jobs.json",
    ):
        _write_json_atomic(root / name, [])
    _write_json_atomic(root / "tables" / "index.json", [])


def _session_state_for_save() -> dict[str, Any]:
    from imajin.agent import state

    return state.snapshot_session_state()


def _file_records_for_save(root: Path) -> list[dict[str, Any]]:
    from imajin.agent import state

    records: list[dict[str, Any]] = []
    for rec in state.list_files():
        p = Path(rec.get("path", "")).expanduser()
        exists = p.exists()
        stat = p.stat() if exists else None
        out = dict(rec)
        out["original_path"] = str(p)
        out["relative_path"] = _relative_path(root, p)
        out["exists"] = exists
        out["size_bytes"] = int(stat.st_size) if stat else None
        out["modified_time"] = (
            datetime.fromtimestamp(stat.st_mtime, UTC).isoformat() if stat else None
        )
        records.append(out)
    return records


def _file_records_for_load(
    root: Path, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for rec in records:
        resolved = _resolve_project_file_path(root, rec)
        exists = resolved.exists()
        load_status = "unloaded" if exists else "missing"
        if not exists:
            warnings.append(f"missing raw file for {rec.get('file_id')}: {resolved}")
        elif rec.get("size_bytes") is not None:
            current_size = resolved.stat().st_size
            if int(rec["size_bytes"]) != int(current_size):
                load_status = "changed"
                warnings.append(f"raw file size changed for {rec.get('file_id')}: {resolved}")
        item = dict(rec)
        item["path"] = str(resolved)
        item["load_status"] = load_status
        out.append(item)
    return out, warnings


def _resolve_project_file_path(root: Path, rec: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if rec.get("original_path"):
        candidates.append(Path(str(rec["original_path"])).expanduser())
    elif rec.get("path"):
        candidates.append(Path(str(rec["path"])).expanduser())
    if rec.get("relative_path"):
        candidates.append((root / str(rec["relative_path"])).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else root / str(rec.get("original_name") or "")


def _save_tables(root: Path, table_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from imajin.agent import state

    records: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in table_specs:
        name = str(item["name"])
        df = state.get_table(name)
        stem = _unique_stem(_slugify(name), used)
        table_path = root / "tables" / f"{stem}.parquet"
        fmt = "parquet"
        try:
            df.to_parquet(table_path, index=False)
        except Exception:
            table_path = root / "tables" / f"{stem}.csv"
            fmt = "csv"
            df.to_csv(table_path, index=False)
        records.append(
            {
                "name": name,
                "path": str(table_path.relative_to(root)),
                "format": fmt,
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "columns": [str(c) for c in df.columns],
                "spec": item.get("spec") or {},
            }
        )
    return records


def _load_tables(root: Path) -> int:
    from imajin.agent import state

    state.reset_tables()
    records = _read_json(root / "tables" / "index.json", default=[])
    count = 0
    for rec in records:
        p = (root / str(rec["path"])).resolve()
        if not p.exists():
            continue
        if rec.get("format") == "csv" or p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        state.set_table(str(rec["name"]), df, spec=dict(rec.get("spec") or {}))
        count += 1
    return count


def _jobs_for_save() -> list[dict[str, Any]]:
    from imajin.agent.execution import get_execution_service

    return [_json_safe(asdict(job)) for job in get_execution_service().list_jobs()]


def _load_jobs(root: Path) -> int:
    from imajin.agent.execution import Job, get_execution_service

    records = _read_json(root / "jobs.json", default=[])
    jobs = [Job(**rec) for rec in records]
    get_execution_service().replace_jobs(jobs)
    return len(jobs)


def _copy_current_provenance(root: Path) -> list[dict[str, Any]]:
    from imajin.agent import provenance

    src = provenance.current_session_path()
    if src is None or not src.exists():
        return []
    dst = root / "provenance" / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return [{"path": str(dst.relative_to(root)), "session_id": provenance.current_session_id()}]


def _project_result(root: Path, metadata: ProjectMetadata, *, created: bool) -> dict[str, Any]:
    return {
        "project_id": metadata.project_id,
        "name": metadata.name,
        "path": str(root),
        "schema_version": metadata.schema_version,
        "created": created,
    }


def _require_project() -> ProjectContext:
    if _CURRENT_PROJECT is None:
        raise RuntimeError("No current project. Call create_project(path) first.")
    return _CURRENT_PROJECT


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
            f.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _relative_path(root: Path, path: Path) -> str | None:
    try:
        return os.path.relpath(path.resolve(), root)
    except OSError:
        return None


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return text or "table"


def _unique_stem(stem: str, used: set[str]) -> str:
    candidate = stem
    i = 2
    while candidate in used:
        candidate = f"{stem}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _now() -> str:
    return datetime.now(UTC).isoformat()
