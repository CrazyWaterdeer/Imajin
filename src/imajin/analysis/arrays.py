from __future__ import annotations

from typing import Any

import numpy as np


def materialize_array(data: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return an in-memory numpy array from numpy-like or dask-like data."""

    if hasattr(data, "compute"):
        data = data.compute()
    if dtype is None:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


def metadata_axes_without_channel(
    metadata: dict[str, Any] | None,
    ndim: int,
) -> str | None:
    """Return metadata axes aligned to `ndim`, excluding channel axes."""

    axes = metadata.get("axes") if isinstance(metadata, dict) else None
    if not isinstance(axes, str):
        return None
    layer_axes = axes.replace("C", "")
    if len(layer_axes) != ndim:
        return None
    return layer_axes


def default_axes(ndim: int, *, default_3d: str = "ZYX") -> str:
    if ndim == 4:
        return "TZYX"
    if ndim == 3:
        return default_3d
    if ndim == 2:
        return "YX"
    return "".join(f"A{i}" for i in range(ndim))


def layer_axes_from_metadata(
    metadata: dict[str, Any] | None,
    ndim: int,
    *,
    default_3d: str = "ZYX",
) -> str:
    return metadata_axes_without_channel(metadata, ndim) or default_axes(
        ndim,
        default_3d=default_3d,
    )
