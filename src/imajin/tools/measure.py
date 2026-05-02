from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import (
    get_table,
    get_table_entry,
    put_table,
    update_table,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool

_DEFAULT_PROPS = [
    "label",
    "area",
    "centroid",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
]
_TIME_PROPS = ["label", "area", "mean_intensity", "max_intensity", "min_intensity"]


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _voxel_scale(scale: tuple[float, ...] | None, ndim: int) -> tuple[float, ...] | None:
    """Return scale aligned to the given dimensionality, or None if scale is
    missing/trivial (all 1.0)."""
    if not scale:
        return None
    s = tuple(float(v) for v in scale[:ndim])
    if len(s) != ndim:
        return None
    if all(abs(v - 1.0) < 1e-9 for v in s):
        return None
    return s


def _add_physical_columns(
    df: pd.DataFrame,
    scale: tuple[float, ...] | None,
    ndim: int,
) -> pd.DataFrame:
    """Add physical-unit columns alongside the pixel/voxel columns produced by
    regionprops. Behaves additively — original columns are preserved."""

    if df.empty:
        return df

    # Always alias raw pixel columns under explicit names so downstream
    # consumers can rely on stable headers.
    if "area" in df.columns:
        if ndim == 3:
            df = df.assign(volume_voxels=df["area"].astype(float))
        else:
            df = df.assign(area_px=df["area"].astype(float))

    s = _voxel_scale(scale, ndim)
    if s is None:
        return df

    if ndim == 2 and "area" in df.columns:
        sy, sx = s
        df = df.assign(area_um2=df["area"].astype(float) * float(sy * sx))
    elif ndim == 3 and "area" in df.columns:
        sz, sy, sx = s
        df = df.assign(volume_um3=df["area"].astype(float) * float(sz * sy * sx))

    centroid_axes = ("z", "y", "x") if ndim == 3 else ("y", "x")
    for i, axis in enumerate(centroid_axes):
        col = f"centroid-{i}"
        if col in df.columns:
            df[f"centroid_{axis}_um"] = df[col].astype(float) * float(s[i])

    return df


def _layer_axes(layer: Any, ndim: int) -> str:
    md = getattr(layer, "metadata", {}) or {}
    axes = md.get("axes") if isinstance(md, dict) else None
    if isinstance(axes, str):
        layer_axes = axes.replace("C", "")
        if len(layer_axes) == ndim:
            return layer_axes
    if ndim == 4:
        return "TZYX"
    if ndim == 3:
        return "TYX"
    if ndim == 2:
        return "YX"
    return "".join(f"A{i}" for i in range(ndim))


def _resolve_time_axis(layer: Any, image_ndim: int, time_axis: int | str | None) -> int:
    axes = _layer_axes(layer, image_ndim)
    if time_axis is None:
        if "T" in axes:
            return axes.index("T")
        return 0
    if isinstance(time_axis, int):
        idx = time_axis if time_axis >= 0 else image_ndim + time_axis
        if idx < 0 or idx >= image_ndim:
            raise ValueError(f"time_axis {time_axis} out of range for {image_ndim}-D image")
        return idx
    code = time_axis.upper()
    if len(code) != 1:
        raise ValueError(f"time_axis must be an axis code or integer, got {time_axis!r}")
    if code not in axes:
        raise ValueError(f"axis {time_axis!r} not found in image axes {axes!r}")
    return axes.index(code)


def _run_regionprops(
    label_arr: np.ndarray,
    image_arrs: list[np.ndarray],
    channel_names: list[str],
    properties: list[str],
) -> pd.DataFrame:
    from skimage.measure import regionprops_table

    base_props = [p for p in properties if p != "label"]
    if not base_props:
        base_props = ["area"]

    frames: list[pd.DataFrame] = []
    for i, (img, cname) in enumerate(zip(image_arrs, channel_names)):
        table = regionprops_table(
            label_arr, intensity_image=img, properties=["label", *base_props]
        )
        df = pd.DataFrame(table)
        if i == 0:
            keep_geometry = df[
                [c for c in df.columns if c == "label" or c.startswith("centroid") or c == "area"]
            ]
            df_other = df[
                [
                    c
                    for c in df.columns
                    if c != "label" and not c.startswith("centroid") and c != "area"
                ]
            ]
            df_other = df_other.add_suffix(f"_{cname}")
            frames.append(pd.concat([keep_geometry, df_other], axis=1))
        else:
            df_other = df.drop(
                columns=[c for c in df.columns if c == "label" or c.startswith("centroid") or c == "area"]
            )
            df_other = df_other.add_suffix(f"_{cname}")
            frames.append(df_other)

    out = pd.concat(frames, axis=1)
    return out


def _resolve_time_interval(image_layer: Any) -> float | None:
    """Pull a per-frame time interval (seconds) from layer metadata when available."""
    md = getattr(image_layer, "metadata", None) or {}
    if not isinstance(md, dict):
        return None
    for key in ("time_interval_s", "time_interval", "frame_interval_s", "frame_interval"):
        value = md.get(key)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0 and np.isfinite(v):
            return v
    return None


def _run_regionprops_over_time(
    label_arr: np.ndarray,
    image_arr: np.ndarray,
    image_layer_name: str,
    time_axis: int,
    properties: list[str],
    time_interval_s: float | None = None,
) -> pd.DataFrame:
    from skimage.measure import regionprops_table

    base_props = [p for p in properties if p != "label"]
    if not base_props:
        base_props = ["mean_intensity"]

    image_time_shape = tuple(
        s for i, s in enumerate(image_arr.shape) if i != time_axis
    )
    static_labels = label_arr.shape == image_time_shape
    dynamic_labels = label_arr.shape == image_arr.shape
    if not static_labels and not dynamic_labels:
        raise ValueError(
            "shape mismatch: labels must either match one image frame "
            f"{image_time_shape} or the full time series {image_arr.shape}; "
            f"got labels {label_arr.shape}."
        )

    frames: list[pd.DataFrame] = []
    for t in range(image_arr.shape[time_axis]):
        frame = np.take(image_arr, t, axis=time_axis)
        labels_frame = (
            label_arr if static_labels else np.take(label_arr, t, axis=time_axis)
        )
        if labels_frame.shape != frame.shape:
            raise ValueError(
                f"shape mismatch at time {t}: labels {labels_frame.shape} "
                f"vs image frame {frame.shape}"
            )
        table = regionprops_table(
            labels_frame.astype(np.int32),
            intensity_image=frame,
            properties=["label", *base_props],
        )
        df = pd.DataFrame(table)
        if df.empty:
            continue
        df.insert(0, "time_index", int(t))
        if time_interval_s is not None:
            df.insert(1, "time_s", float(t) * float(time_interval_s))
        # Preserve the legacy `time` column so existing tests/UI keep working.
        df["time"] = int(t)
        df["image_layer"] = image_layer_name
        frames.append(df)

    base_columns = ["time_index"]
    if time_interval_s is not None:
        base_columns.append("time_s")
    base_columns += ["time", "label", "image_layer"]
    if not frames:
        return pd.DataFrame(columns=base_columns)
    return pd.concat(frames, ignore_index=True)


@tool(
    description="Per-cell intensity statistics (regionprops). Provide a Labels layer + "
    "Image layer(s). For multi-channel use measure_intensity multiple times or list of "
    "image_layers. Default properties include label, area, centroid, mean/max/min "
    "intensity. Stores a table referenceable by name.",
    phase="4",
    worker=True,
)
def measure_intensity(
    labels_layer: str,
    image_layers: list[str],
    properties: list[str] | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    if not image_layers:
        raise ValueError("image_layers must be a non-empty list of layer names")

    labels = call_on_main(snapshot_layer, labels_layer)
    label_arr = _materialize(labels.data).astype(np.int32)

    image_arrs: list[np.ndarray] = []
    channel_names: list[str] = []
    for lname in image_layers:
        img_layer = call_on_main(snapshot_layer, lname)
        img = _materialize(img_layer.data)
        if img.shape != label_arr.shape:
            raise ValueError(
                f"shape mismatch: labels {label_arr.shape} vs image {lname} {img.shape}. "
                "All image layers must match the labels layer shape."
            )
        image_arrs.append(img)
        channel_names.append(lname)

    props = properties or list(_DEFAULT_PROPS)
    df = _run_regionprops(label_arr, image_arrs, channel_names, props)
    df = _add_physical_columns(df, labels.scale, label_arr.ndim)

    scale = _voxel_scale(labels.scale, label_arr.ndim)
    spec = {
        "tool": "measure_intensity",
        "labels_layer": labels_layer,
        "image_layers": list(image_layers),
        "properties": props,
        "voxel_scale": list(scale) if scale is not None else None,
        "ndim": int(label_arr.ndim),
    }
    name = call_on_main(
        put_table,
        table_name or f"{labels_layer}_measurements",
        df,
        spec=spec,
    )

    return {
        "table_name": name,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "voxel_scale": list(scale) if scale is not None else None,
        "ndim": int(label_arr.ndim),
        "has_physical_units": scale is not None,
    }


@tool(
    description="Measure ROI/cell intensity over time for live imaging or time-series "
    "confocal data. Provide a Labels layer defining ROIs and one Image layer with a "
    "time axis. Labels may be static (YX/ZYX) or time-varying (TYX/TZYX). Stores a "
    "long-format table with time, label, area, and intensity columns.",
    phase="4",
    worker=True,
)
def measure_intensity_over_time(
    labels_layer: str,
    image_layer: str,
    properties: list[str] | None = None,
    table_name: str | None = None,
    time_axis: int | str | None = None,
) -> dict[str, Any]:
    labels = call_on_main(snapshot_layer, labels_layer)
    image = call_on_main(snapshot_layer, image_layer)
    label_arr = _materialize(labels.data).astype(np.int32)
    image_arr = _materialize(image.data)
    if image_arr.ndim < 3:
        raise ValueError(
            f"measure_intensity_over_time expects a time-series image, "
            f"got shape {image_arr.shape}"
        )

    t_idx = _resolve_time_axis(image, image_arr.ndim, time_axis)
    props = properties or list(_TIME_PROPS)
    interval = _resolve_time_interval(image)
    df = _run_regionprops_over_time(
        label_arr,
        image_arr,
        image_layer_name=image_layer,
        time_axis=t_idx,
        properties=props,
        time_interval_s=interval,
    )

    spec = {
        "tool": "measure_intensity_over_time",
        "labels_layer": labels_layer,
        "image_layer": image_layer,
        "properties": props,
        "time_axis": t_idx,
        "time_interval_s": interval,
    }
    name = call_on_main(
        put_table,
        table_name or f"{labels_layer}_{image_layer}_timecourse",
        df,
        spec=spec,
    )
    labels_seen = int(df["label"].nunique()) if "label" in df.columns else 0
    return {
        "table_name": name,
        "n_rows": int(len(df)),
        "n_labels": labels_seen,
        "n_timepoints": int(image_arr.shape[t_idx]),
        "columns": list(df.columns),
        "time_interval_s": interval,
    }


@tool(
    description="Re-run a previous measurement against the current state of its labels "
    "layer. Use after manually painting / editing masks in napari.",
    phase="4",
    worker=True,
)
def refresh_measurement(table_name: str) -> dict[str, Any]:
    entry = get_table_entry(table_name)
    spec = entry.spec
    if spec.get("tool") != "measure_intensity":
        raise ValueError(
            f"Table {table_name!r} was not produced by measure_intensity; cannot refresh."
        )

    labels = call_on_main(snapshot_layer, spec["labels_layer"])
    label_arr = _materialize(labels.data).astype(np.int32)

    image_arrs: list[np.ndarray] = []
    channel_names: list[str] = []
    for lname in spec["image_layers"]:
        img_layer = call_on_main(snapshot_layer, lname)
        img = _materialize(img_layer.data)
        image_arrs.append(img)
        channel_names.append(lname)

    df = _run_regionprops(label_arr, image_arrs, channel_names, spec["properties"])
    df = _add_physical_columns(df, labels.scale, label_arr.ndim)
    prev_n = len(entry.df)
    call_on_main(update_table, table_name, df)
    return {
        "table_name": table_name,
        "n_rows": int(len(df)),
        "delta_rows": int(len(df)) - int(prev_n),
        "columns": list(df.columns),
    }


@tool(
    description="Filter rows of a table with a pandas-style query expression. "
    "Example expr: 'area > 50 and mean_intensity_GFP > 1000'. Returns new table name.",
    phase="4",
    worker=True,
)
def filter_table(table_name: str, expr: str, new_table_name: str | None = None) -> dict[str, Any]:
    df = get_table(table_name)
    try:
        filtered = df.query(expr)
    except Exception as e:
        raise ValueError(f"filter expression failed: {e}") from e

    spec = {"tool": "filter_table", "source": table_name, "expr": expr}
    name = call_on_main(
        put_table,
        new_table_name or f"{table_name}_filtered",
        filtered.reset_index(drop=True),
        spec=spec,
    )
    return {"table_name": name, "n_rows": int(len(filtered)), "expr": expr}


@tool(
    description="Aggregate a table. op is one of mean/median/sum/count/std/min/max. "
    "Optionally group_by a column. Returns aggregated values inline (small).",
    phase="4",
    worker=True,
)
def summarize_table(
    table_name: str,
    op: str = "mean",
    group_by: str | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    df = get_table(table_name)
    target = df[columns] if columns else df.select_dtypes(include="number")

    if group_by:
        if group_by not in df.columns:
            raise ValueError(f"group_by {group_by!r} not in columns: {list(df.columns)}")
        grouped = df.groupby(group_by)[target.columns.tolist()]
        agg = getattr(grouped, op)()
    else:
        agg = getattr(target, op)()

    if isinstance(agg, pd.Series):
        return {"table_name": table_name, "op": op, "values": agg.to_dict()}
    return {
        "table_name": table_name,
        "op": op,
        "by": group_by,
        "values": agg.to_dict(orient="index"),
    }
