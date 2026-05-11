from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


MissingColumnPolicy = Literal["raise", "empty"]


def finite_numeric_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    missing: MissingColumnPolicy = "raise",
) -> tuple[pd.DataFrame, int]:
    """Return rows where ``value_col`` can be coerced to a finite number."""
    if value_col not in df.columns:
        if missing == "empty":
            return pd.DataFrame(), 0
        raise ValueError(f"value_col {value_col!r} not found in columns: {list(df.columns)}")
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    mask = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    dropped = int(len(out) - int(mask.sum()))
    out[value_col] = values
    return out.loc[mask].reset_index(drop=True), dropped


def infer_time_column(df: pd.DataFrame, time_col: str | None = None) -> str:
    """Resolve an explicit or conventional time column name."""
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"time_col {time_col!r} not found in columns: {list(df.columns)}")
        return time_col
    for candidate in ("time_s", "time_index", "time", "t", "frame"):
        if candidate in df.columns:
            return candidate
    raise ValueError("could not infer a time column; pass time_col explicitly")
