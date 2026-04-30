from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_VIEWER: Any | None = None


@dataclass
class TableEntry:
    df: pd.DataFrame
    spec: dict[str, Any] = field(default_factory=dict)


_TABLES: dict[str, TableEntry] = {}


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
