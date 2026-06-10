"""Reference library for morphometric classification.

A reference library is a CSV with one row per labelled neuron:
``name, label, units_physical, <feature columns…>``. It is built offline from the
user's own labelled traces (see ``append_reference``) and consumed by the matcher.

Pure stdlib + pandas — no new dependency, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


_META_COLUMNS = ("name", "label", "units_physical")


@dataclass
class ReferenceLibrary:
    frame: pd.DataFrame

    @property
    def names(self) -> list[str]:
        return self.frame["name"].astype(str).tolist()

    @property
    def labels(self) -> list[str]:
        return self.frame["label"].astype(str).tolist()

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c not in _META_COLUMNS]

    @property
    def all_physical(self) -> bool:
        """True iff every reference row was measured in physical (micron) units."""
        return bool(self.frame["units_physical"].astype(bool).all())

    def __len__(self) -> int:
        return len(self.frame)


def load_reference_library(path: str | Path) -> ReferenceLibrary:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Reference library not found: {p}. Build one with add_reference_neuron()."
        )
    frame = pd.read_csv(p)
    missing = {"name", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Reference library {p} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Reference library {p} has no rows.")
    if "units_physical" not in frame.columns:
        frame["units_physical"] = False
    return ReferenceLibrary(frame=frame)


def append_reference(
    path: str | Path,
    feature_vector: dict[str, Any],
    *,
    label: str,
    name: str,
) -> ReferenceLibrary:
    """Append one labelled feature vector to a CSV library, creating it if needed.

    ``feature_vector`` is the dict returned by ``extract_feature_vector``. Columns
    are aligned across rows (a pixel-scale row leaves absolute columns blank).
    """
    label = label.strip()
    name = name.strip()
    if not label:
        raise ValueError("label must not be empty")
    if not name:
        raise ValueError("name must not be empty")

    row = {
        "name": name,
        "label": label,
        "units_physical": bool(feature_vector.get("units_physical", False)),
        **feature_vector.get("features", {}),
    }
    new_row = pd.DataFrame([row])

    p = Path(path)
    if p.exists():
        existing = pd.read_csv(p)
        frame = pd.concat([existing, new_row], ignore_index=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        frame = new_row

    frame.to_csv(p, index=False)
    return ReferenceLibrary(frame=frame)
