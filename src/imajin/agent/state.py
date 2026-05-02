from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

_VIEWER: Any | None = None


@dataclass
class TableEntry:
    df: pd.DataFrame
    spec: dict[str, Any] = field(default_factory=dict)


_TABLES: dict[str, TableEntry] = {}


def _slugify(name: str) -> str:
    base = name.rsplit(".", 1)[0]  # strip extension only on the rightmost dot
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base or "file"


@dataclass
class FileRecord:
    file_id: str
    path: str
    original_name: str
    file_type: str | None = None
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    load_status: str = "unloaded"  # "unloaded" | "loaded" | "failed"
    notes: str | None = None


_FILES: dict[str, FileRecord] = {}


def put_file(
    path: str,
    original_name: str,
    file_type: str | None = None,
    metadata_summary: dict[str, Any] | None = None,
    notes: str | None = None,
    load_status: str = "unloaded",
) -> str:
    base = _slugify(original_name)
    file_id = base
    n = 2
    while file_id in _FILES:
        file_id = f"{base}_{n}"
        n += 1
    _FILES[file_id] = FileRecord(
        file_id=file_id,
        path=path,
        original_name=original_name,
        file_type=file_type,
        metadata_summary=dict(metadata_summary or {}),
        notes=notes,
        load_status=load_status,
    )
    return file_id


def get_file(file_id: str) -> FileRecord:
    if file_id not in _FILES:
        raise KeyError(
            f"File id {file_id!r} not found. Available: {list(_FILES)}"
        )
    return _FILES[file_id]


def list_files() -> list[dict[str, Any]]:
    return [asdict(rec) for rec in _FILES.values()]


def update_file_status(file_id: str, status: str, notes: str | None = None) -> None:
    rec = get_file(file_id)
    rec.load_status = status
    if notes is not None:
        rec.notes = notes


def reset_files() -> None:
    _FILES.clear()


def reset_recipes() -> None:
    """Stub — real implementation in Task 3."""


def reset_runs() -> None:
    """Stub — real implementation in Task 4."""


def set_viewer(v: Any) -> None:
    global _VIEWER
    _VIEWER = v


def get_viewer() -> Any:
    if _VIEWER is None:
        raise RuntimeError(
            "No napari viewer registered. Call set_viewer(viewer) at startup, "
            "or pass arrays directly when calling tools from a script."
        )
    return _VIEWER


def viewer_or_none() -> Any | None:
    return _VIEWER


def get_layer(name: str) -> Any:
    viewer = get_viewer()
    try:
        return viewer.layers[name]
    except KeyError as e:
        names = [L.name for L in viewer.layers]
        raise KeyError(f"Layer {name!r} not found. Available: {names}") from e


def get_table(name: str) -> pd.DataFrame:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found. Available: {list(_TABLES)}")
    return _TABLES[name].df


def get_table_entry(name: str) -> TableEntry:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found. Available: {list(_TABLES)}")
    return _TABLES[name]


def put_table(
    name: str, df: pd.DataFrame, spec: dict[str, Any] | None = None
) -> str:
    base = name
    i = 1
    while name in _TABLES:
        name = f"{base}_{i}"
        i += 1
    _TABLES[name] = TableEntry(df=df, spec=dict(spec or {}))
    _emit_tables_changed()
    return name


def update_table(name: str, df: pd.DataFrame) -> None:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found")
    _TABLES[name].df = df
    _emit_tables_changed()


def list_tables() -> list[str]:
    return list(_TABLES)


def reset_tables() -> None:
    _TABLES.clear()
    _emit_tables_changed()


_TABLE_LISTENERS: list[Any] = []


def on_tables_changed(callback: Any) -> None:
    if callback not in _TABLE_LISTENERS:
        _TABLE_LISTENERS.append(callback)


def _emit_tables_changed() -> None:
    for cb in list(_TABLE_LISTENERS):
        try:
            cb()
        except Exception:
            pass
