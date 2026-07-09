from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import (
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


_materialize = materialize_array  # shared: analysis.arrays.materialize_array


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


def _add_region_column(df: pd.DataFrame, label_names: Any) -> pd.DataFrame:
    """Map each row's integer ``label`` to a name via a ``{int: str}`` dict carried on the
    labels layer metadata (``label_names``), added as a ``region`` column. Best-effort:
    keys that don't coerce to int are skipped; unmapped labels become NaN, and rows are
    never dropped. No-op when ``label_names`` is not a (non-empty) dict or the frame has no
    ``label`` column. Lets a partition layer ({1: "inside", 2: "outside"}) self-describe its
    measurement rows without a fragile caller-side post-step."""
    if not isinstance(label_names, dict) or "label" not in df.columns:
        return df
    mapping: dict[int, str] = {}
    for key, value in label_names.items():
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    if not mapping:
        return df
    region = df["label"].map(mapping)
    if "region" in df.columns:
        df["region"] = region
    else:
        df.insert(df.columns.get_loc("label") + 1, "region", region)
    return df


def _layer_axes(layer: Any, ndim: int) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return layer_axes_from_metadata(md, ndim, default_3d="ZYX")


def _resolve_time_axis(layer: Any, image_ndim: int, time_axis: int | str | None) -> int:
    axes = _layer_axes(layer, image_ndim)
    if time_axis is None:
        if "T" in axes:
            return axes.index("T")
        raise ValueError(
            f"image layer axes {axes!r} do not include a time axis. Reload with "
            "metadata axes containing 'T' or pass time_axis explicitly."
        )
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


# Intensity properties whose multichannel regionprops output is one column per
# channel ({prop}-{i}); the fast path renames these to {prop}_{channel}.
_INTENSITY_SCALAR_PROPS = frozenset({
    "mean_intensity", "max_intensity", "min_intensity",
    "intensity_mean", "intensity_max", "intensity_min", "intensity_std",
})
# Geometry properties the per-channel path keeps once (unsuffixed). Together with
# the scalar-intensity set they define when the single-pass fast path is exact.
_GEOMETRY_KEPT_PROPS = frozenset({"area", "centroid"})


def _run_regionprops_per_channel(
    label_arr: np.ndarray,
    image_arrs: list[np.ndarray],
    channel_names: list[str],
    base_props: list[str],
) -> pd.DataFrame:
    """One regionprops pass per channel: geometry (label/area/centroid) kept once,
    every other prop suffixed ``_{channel}``. The exact fallback for uncommon
    property sets the single-pass fast path can't name unambiguously."""
    from skimage.measure import regionprops_table

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

    return pd.concat(frames, axis=1)


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

    # Fast path: one region pass over a single multichannel intensity image
    # (channels stacked on the trailing axis) instead of N passes. Used only when
    # every property is a known kept-geometry or scalar-intensity prop, so the
    # resulting column set and order are byte-identical to the per-channel path;
    # anything else (extra geometry props, weighted/multi-index intensity) falls
    # back to the loop.
    fast_ok = all(
        p in _GEOMETRY_KEPT_PROPS or p in _INTENSITY_SCALAR_PROPS for p in base_props
    )
    if not fast_ok:
        return _run_regionprops_per_channel(label_arr, image_arrs, channel_names, base_props)

    intensity = np.stack(image_arrs, axis=-1)
    table = regionprops_table(
        label_arr, intensity_image=intensity, properties=["label", *base_props]
    )
    df = pd.DataFrame(table)

    geometry_props = [p for p in base_props if p in _GEOMETRY_KEPT_PROPS]
    intensity_props = [p for p in base_props if p in _INTENSITY_SCALAR_PROPS]

    # Preserve the per-channel path's column order: label, geometry (once), then
    # intensity grouped by channel.
    result: dict[str, Any] = {"label": df["label"]}
    for prop in geometry_props:
        for col in df.columns:
            if col == prop or col.startswith(f"{prop}-"):
                result[col] = df[col]
    for i, cname in enumerate(channel_names):
        for prop in intensity_props:
            result[f"{prop}_{cname}"] = df[f"{prop}-{i}"]
    return pd.DataFrame(result)


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
    df = _add_region_column(df, (labels.metadata or {}).get("label_names"))

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
    description="Project a z-stack image first, then measure ROI/cell intensity from "
    "2D Labels on the projected image. Default projection='mean' because average "
    "projection is the standard for intensity comparison workflows; use projection="
    "'max' for representative morphology figures.",
    phase="4",
    worker=True,
)
def measure_projected_intensity(
    labels_layer: str,
    image_layer: str,
    projection: str = "mean",
    axis: int | str = "z",
    properties: list[str] | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    from imajin.tools import view

    mode = projection.lower().strip()
    if mode in {"mean", "avg", "average"}:
        proj = view.average_projection(image_layer, axis=axis)
    elif mode in {"max", "mip", "maximum"}:
        proj = view.max_projection(image_layer, axis=axis)
    else:
        raise ValueError("projection must be mean or max")

    measured = measure_intensity(
        labels_layer=labels_layer,
        image_layers=[proj["new_layer"]],
        properties=properties,
        table_name=table_name,
    )
    return {
        **measured,
        "projection": "mean" if mode in {"mean", "avg", "average"} else "max",
        "projected_layer": proj["new_layer"],
        "source_image_layer": image_layer,
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
    df = _add_region_column(df, (labels.metadata or {}).get("label_names"))
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


@tool(
    description="Concatenate several registered tables row-wise into one combined table, "
    "tagging each source's rows with a replicate/specimen id in `label_column` "
    "(default 'sample_name'). Use this to merge the per-file tables of the manual "
    "per-image path — e.g. inside/outside coloc tables for rep1/rep2/rep3 — into a "
    "single table that compare_groups (paired), summarize_experiment, the plot_* tools, "
    "and export_table can consume. Pass `labels` to name each replicate (one per table, "
    "same order); omit it to label rows by their source table name. Columns are unioned "
    "across sources (cells missing in a source become NaN). Returns the new table name "
    "and per-source row counts.",
    phase="4",
    worker=True,
)
def combine_tables(
    table_names: list[str],
    new_table_name: str | None = None,
    label_column: str = "sample_name",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if not table_names:
        raise ValueError("combine_tables needs at least one table name.")
    if labels is not None and len(labels) != len(table_names):
        raise ValueError(
            f"labels has {len(labels)} entries but table_names has {len(table_names)}; "
            "pass exactly one label per table (same order)."
        )

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for i, tname in enumerate(table_names):
        df = get_table(tname).copy()  # raises KeyError (with available names) if missing
        label = labels[i] if labels is not None else tname
        if label_column in df.columns and labels is None:
            raise ValueError(
                f"table {tname!r} already has a {label_column!r} column; pass explicit "
                "labels=[...] (one per table) to set replicate ids without clobbering it, "
                "or choose a different label_column."
            )
        df[label_column] = label
        frames.append(df)
        sources.append({"table": tname, "label": label, "n_rows": int(len(df))})

    combined = pd.concat(frames, ignore_index=True, sort=False)

    name = call_on_main(
        put_table,
        new_table_name or "combined",
        combined,
        spec={
            "tool": "combine_tables",
            "sources": list(table_names),
            "label_column": label_column,
        },
    )
    shared = set.intersection(*(set(f.columns) for f in frames))
    return {
        "table_name": name,
        "n_rows": int(len(combined)),
        "n_sources": len(table_names),
        "label_column": label_column,
        "labels": [s["label"] for s in sources],
        "sources": sources,
        "columns": list(combined.columns),
        "columns_not_in_all_sources": [c for c in combined.columns if c not in shared],
    }


@tool(
    description="Merge several columns into one by taking the first non-null value per row "
    "(coalesce). Built for the manual per-file path: combining per-file tables leaves one "
    "SPARSE intensity column per file (mean_intensity_<file1>, mean_intensity_<file2>, …), each "
    "filled only for its file's rows, which compare_groups / plot_group_distribution cannot use. "
    "Coalescing them into a single `mean_intensity` column fixes that. Select the columns "
    "explicitly, or pass a `prefix` to match them (e.g. prefix='mean_intensity_'). Updates the "
    "table in place unless new_table_name is given; source columns are dropped by default.",
    phase="4",
    worker=True,
)
def coalesce_columns(
    table_name: str,
    into: str,
    columns: list[str] | None = None,
    prefix: str | None = None,
    drop_sources: bool = True,
    new_table_name: str | None = None,
) -> dict[str, Any]:
    df = get_table(table_name).copy()
    if prefix:
        selected = [c for c in df.columns if c.startswith(prefix)]
    else:
        selected = list(columns or [])
    if not selected:
        raise ValueError(
            "coalesce_columns selected no columns: pass `columns` explicitly or a `prefix` "
            "that matches existing column names."
        )
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in table {table_name!r}: {missing}")

    src = df[selected]
    # first non-null across the selected columns, left to right
    coalesced = src.bfill(axis=1).iloc[:, 0]
    ambiguous = int((src.notna().sum(axis=1) > 1).sum())
    df[into] = coalesced
    if drop_sources:
        df = df.drop(columns=[c for c in selected if c != into])

    spec = {"tool": "coalesce_columns", "into": into, "sources": selected}
    if new_table_name:
        name = call_on_main(put_table, new_table_name, df, spec)
    else:
        call_on_main(update_table, table_name, df)
        name = table_name
    return {
        "table_name": name,
        "into": into,
        "n_sources": len(selected),
        "sources": selected,
        "n_filled": int(coalesced.notna().sum()),
        "n_rows": int(len(df)),
        "ambiguous_rows": ambiguous,
        "columns": list(df.columns),
    }


@tool(
    description="Add or overwrite a column by mapping an existing column through a user-provided "
    "dict — e.g. assign a `group` from `sample_name` ({'mF_rectum_1': 'mated', 'vF_rectum_1': "
    "'virgin', …}) so compare_groups / plots can split by group. The mapping is explicit and "
    "user-confirmed; nothing is inferred from the names. Unmapped values receive `default` "
    "(None leaves them blank). Updates the table in place unless new_table_name is given.",
    phase="4",
)
def map_column(
    table_name: str,
    from_col: str,
    mapping: dict[str, str],
    into: str = "group",
    default: str | None = None,
    new_table_name: str | None = None,
) -> dict[str, Any]:
    df = get_table(table_name).copy()
    if from_col not in df.columns:
        raise ValueError(
            f"from_col {from_col!r} not in table {table_name!r}: {list(df.columns)}"
        )
    keys = df[from_col].astype(str)
    df[into] = keys.map(lambda v: mapping.get(v, default))
    unmapped = sorted(set(keys.unique()) - set(mapping))

    spec = {"tool": "map_column", "from_col": from_col, "into": into}
    if new_table_name:
        name = call_on_main(put_table, new_table_name, df, spec)
    else:
        call_on_main(update_table, table_name, df)
        name = table_name
    return {
        "table_name": name,
        "into": into,
        "from_col": from_col,
        "n_mapped": int(df[into].notna().sum()),
        "n_rows": int(len(df)),
        "unmapped_values": unmapped,
        "distinct_groups": sorted(set(df[into].dropna().astype(str))),
    }


@tool(
    description="Keep one representative row per group — e.g. the largest object per sample. "
    "Groups by `group_by` (default 'sample_name') and within each group keeps the single row "
    "with the max (or min) of `by`. Use `by='area', keep='max'` to keep each sample's main "
    "region and drop small debris objects before a per-sample group comparison. Updates the "
    "table in place unless new_table_name is given.",
    phase="4",
    worker=True,
)
def select_representative_rows(
    table_name: str,
    by: str,
    group_by: str = "sample_name",
    keep: Literal["max", "min"] = "max",
    new_table_name: str | None = None,
) -> dict[str, Any]:
    df = get_table(table_name).copy()
    for col in (by, group_by):
        if col not in df.columns:
            raise ValueError(
                f"column {col!r} not in table {table_name!r}: {list(df.columns)}"
            )
    n_before = len(df)
    order = pd.to_numeric(df[by], errors="coerce")
    df = df.assign(_ord=order).sort_values(
        "_ord", ascending=(keep == "min"), na_position="last"
    )
    reduced = (
        df.drop_duplicates(subset=[group_by], keep="first")
        .drop(columns="_ord")
        .sort_index()
        .reset_index(drop=True)
    )

    spec = {"tool": "select_representative_rows", "by": by, "group_by": group_by, "keep": keep}
    if new_table_name:
        name = call_on_main(put_table, new_table_name, reduced, spec)
    else:
        call_on_main(update_table, table_name, reduced)
        name = table_name
    return {
        "table_name": name,
        "by": by,
        "group_by": group_by,
        "keep": keep,
        "n_rows_before": int(n_before),
        "n_rows_after": int(len(reduced)),
        "n_groups": int(reduced[group_by].nunique()),
    }
