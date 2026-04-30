from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.state import (
    get_layer,
    get_table,
    get_table_entry,
    put_table,
    update_table,
)
from imajin.tools.registry import tool

_DEFAULT_PROPS = ["label", "area", "centroid", "mean_intensity", "max_intensity", "min_intensity"]


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


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


@tool(
    description="Per-cell intensity statistics (regionprops). Provide a Labels layer + "
    "Image layer(s). For multi-channel use measure_intensity multiple times or list of "
    "image_layers. Default properties include label, area, centroid, mean/max/min "
    "intensity. Stores a table referenceable by name.",
    phase="4",
)
def measure_intensity(
    labels_layer: str,
    image_layers: list[str],
    properties: list[str] | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    if not image_layers:
        raise ValueError("image_layers must be a non-empty list of layer names")

    labels = get_layer(labels_layer)
    label_arr = _materialize(labels.data).astype(np.int32)

    image_arrs: list[np.ndarray] = []
    channel_names: list[str] = []
    for lname in image_layers:
        img_layer = get_layer(lname)
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

    spec = {
        "tool": "measure_intensity",
        "labels_layer": labels_layer,
        "image_layers": list(image_layers),
        "properties": props,
    }
    name = put_table(table_name or f"{labels_layer}_measurements", df, spec=spec)

    return {
        "table_name": name,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }


@tool(
    description="Re-run a previous measurement against the current state of its labels "
    "layer. Use after manually painting / editing masks in napari.",
    phase="4",
)
def refresh_measurement(table_name: str) -> dict[str, Any]:
    entry = get_table_entry(table_name)
    spec = entry.spec
    if spec.get("tool") != "measure_intensity":
        raise ValueError(
            f"Table {table_name!r} was not produced by measure_intensity; cannot refresh."
        )

    labels = get_layer(spec["labels_layer"])
    label_arr = _materialize(labels.data).astype(np.int32)

    image_arrs: list[np.ndarray] = []
    channel_names: list[str] = []
    for lname in spec["image_layers"]:
        img_layer = get_layer(lname)
        img = _materialize(img_layer.data)
        image_arrs.append(img)
        channel_names.append(lname)

    df = _run_regionprops(label_arr, image_arrs, channel_names, spec["properties"])
    prev_n = len(entry.df)
    update_table(table_name, df)
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
)
def filter_table(table_name: str, expr: str, new_table_name: str | None = None) -> dict[str, Any]:
    df = get_table(table_name)
    try:
        filtered = df.query(expr)
    except Exception as e:
        raise ValueError(f"filter expression failed: {e}") from e

    spec = {"tool": "filter_table", "source": table_name, "expr": expr}
    name = put_table(new_table_name or f"{table_name}_filtered", filtered.reset_index(drop=True), spec=spec)
    return {"table_name": name, "n_rows": int(len(filtered)), "expr": expr}


@tool(
    description="Aggregate a table. op is one of mean/median/sum/count/std/min/max. "
    "Optionally group_by a column. Returns aggregated values inline (small).",
    phase="4",
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
