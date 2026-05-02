from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin.agent import state
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools.napari_ops import add_image_from_worker, snapshot_layer
from imajin.tools.registry import tool

_QC_STATUSES = {"pass", "warning", "fail", "not_checked"}


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _record(source: str, warnings: list[str], metrics: dict[str, Any]) -> dict[str, Any]:
    status = "fail" if metrics.get("failed") else ("warning" if warnings else "pass")
    state.put_qc_record(
        source=source,
        status=status,
        warnings=warnings,
        metrics=_json_value(metrics),
    )
    return {
        "source": source,
        "status": status,
        "warnings": warnings,
        "metrics": _json_value(metrics),
    }


def _scale_is_missing(scale: tuple[float, ...] | None, ndim: int) -> bool:
    if not scale or len(scale) < ndim:
        return True
    s = tuple(float(v) for v in scale[:ndim])
    return all(abs(v - 1.0) < 1e-9 for v in s)


def _border_touching_labels(labels: np.ndarray) -> set[int]:
    border: list[np.ndarray] = []
    for axis, size in enumerate(labels.shape):
        if size == 0:
            continue
        border.append(np.take(labels, 0, axis=axis))
        border.append(np.take(labels, size - 1, axis=axis))
    if not border:
        return set()
    values = np.concatenate([np.ravel(b) for b in border])
    return {int(v) for v in np.unique(values) if int(v) > 0}


def _saturation_fraction(image: np.ndarray) -> tuple[float, bool]:
    if image.size == 0:
        return 0.0, False
    if np.issubdtype(image.dtype, np.integer):
        threshold = np.iinfo(image.dtype).max
        return float(np.count_nonzero(image >= threshold) / image.size), True
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, False
    max_value = float(np.max(finite))
    if max_value <= 1.0:
        return float(np.count_nonzero(finite >= 0.999) / finite.size), True
    return float(np.count_nonzero(finite >= max_value) / finite.size), False


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def _intensity_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in _numeric_columns(df) if "intensity" in str(c).lower()]


def _size_columns(df: pd.DataFrame) -> list[str]:
    names = {
        "area",
        "area_px",
        "area_um2",
        "volume_voxels",
        "volume_um3",
    }
    return [c for c in _numeric_columns(df) if str(c).lower() in names]


def _time_column(df: pd.DataFrame) -> str | None:
    for col in ("time_index", "time", "time_s", "t", "frame"):
        if col in df.columns:
            return col
    return None


def _centroid_from_row(row: pd.Series, labels_layer: str | None) -> tuple[float, ...] | None:
    raw_cols = sorted(
        [c for c in row.index if str(c).startswith("centroid-")],
        key=lambda c: int(str(c).split("-", 1)[1]),
    )
    if raw_cols:
        coords = [float(row[c]) for c in raw_cols]
        if labels_layer:
            try:
                layer = state.get_layer(labels_layer)
                scale = tuple(float(v) for v in getattr(layer, "scale", ()))
            except Exception:
                scale = ()
            if len(scale) >= len(coords):
                coords = [c * scale[i] for i, c in enumerate(coords)]
        return tuple(coords)

    physical_sets = [
        ("centroid_z_um", "centroid_y_um", "centroid_x_um"),
        ("centroid_y_um", "centroid_x_um"),
        ("z_um", "y_um", "x_um"),
        ("y_um", "x_um"),
    ]
    for cols in physical_sets:
        if all(c in row.index for c in cols):
            return tuple(float(row[c]) for c in cols)
    return None


@tool(
    description="Compute QC metrics for a Labels layer. Optionally provide a matching "
    "image layer to check intensity saturation. Stores a QC record for later reports.",
    phase="6",
    worker=True,
)
def compute_segmentation_qc(
    labels_layer: str,
    image_layer: str | None = None,
) -> dict[str, Any]:
    labels_snapshot = call_on_main(snapshot_layer, labels_layer)
    labels = _materialize(labels_snapshot.data).astype(np.int64, copy=False)

    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "kind": "segmentation",
        "labels_layer": labels_layer,
        "shape": tuple(int(s) for s in labels.shape),
        "dtype": str(labels.dtype),
        "failed": False,
    }

    positive = labels[labels > 0]
    if positive.size == 0:
        warnings.append("No labeled objects were found.")
        metrics.update(
            {
                "n_objects": 0,
                "object_size_min": 0,
                "object_size_median": 0.0,
                "object_size_max": 0,
                "failed": True,
            }
        )
        return _record(labels_layer, warnings, metrics)

    unique, counts = np.unique(positive, return_counts=True)
    sizes = counts.astype(float)
    median = float(np.median(sizes))
    tiny_fraction = float(np.mean(sizes < max(1.0, median * 0.25))) if median else 0.0
    large_fraction = float(np.mean(sizes > median * 4.0)) if median else 0.0
    border_labels = _border_touching_labels(labels)
    border_fraction = float(len(border_labels) / len(unique))

    metrics.update(
        {
            "n_objects": int(len(unique)),
            "object_size_min": int(np.min(counts)),
            "object_size_median": median,
            "object_size_max": int(np.max(counts)),
            "object_size_mean": float(np.mean(sizes)),
            "tiny_object_fraction": tiny_fraction,
            "large_object_fraction": large_fraction,
            "border_touching_labels": sorted(border_labels),
            "border_touching_fraction": border_fraction,
            "scale": tuple(float(v) for v in labels_snapshot.scale),
        }
    )

    if _scale_is_missing(labels_snapshot.scale, labels.ndim):
        warnings.append("Layer scale is missing or unit-valued; physical sizes may be unavailable.")
    if tiny_fraction > 0.25:
        warnings.append("Many labeled objects are much smaller than the median object size.")
    if large_fraction > 0.10:
        warnings.append("Some labeled objects are much larger than the median object size.")
    if border_fraction > 0.10:
        warnings.append("Some labeled objects touch the image border.")

    if image_layer is not None:
        image_snapshot = call_on_main(snapshot_layer, image_layer)
        image = _materialize(image_snapshot.data)
        if image.shape != labels.shape:
            raise ValueError(
                f"shape mismatch: labels {labels.shape} vs image {image_layer} {image.shape}"
            )
        saturation, interpretable = _saturation_fraction(image)
        metrics["image_layer"] = image_layer
        metrics["saturation_fraction"] = saturation
        metrics["saturation_interpretable"] = bool(interpretable)
        if interpretable and saturation > 0.001:
            warnings.append("Image has saturated pixels; intensity measurements may be clipped.")

    return _record(labels_layer, warnings, metrics)


@tool(
    description="Compute QC metrics for a per-cell measurement table. Checks empty "
    "tables, missing intensity/size columns, null values, all-zero intensity, and "
    "broad object-size distributions. Stores a QC record.",
    phase="6",
    worker=True,
)
def compute_measurement_qc(table_name: str) -> dict[str, Any]:
    df = state.get_table(table_name)
    warnings: list[str] = []
    intensity_cols = _intensity_columns(df)
    size_cols = _size_columns(df)
    metrics: dict[str, Any] = {
        "kind": "measurement",
        "table_name": table_name,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "intensity_columns": intensity_cols,
        "size_columns": size_cols,
        "failed": False,
    }

    if df.empty:
        warnings.append("Measurement table is empty.")
        metrics["failed"] = True
        return _record(table_name, warnings, metrics)

    if "label" in df.columns:
        metrics["n_labels"] = int(df["label"].nunique(dropna=True))
        duplicate_labels = int(df["label"].duplicated().sum())
        metrics["duplicate_label_rows"] = duplicate_labels
        if duplicate_labels:
            warnings.append("Measurement table has duplicate label rows.")
    else:
        warnings.append("Measurement table has no label column.")

    null_fraction = float(df.isna().sum().sum() / max(1, df.size))
    metrics["null_fraction"] = null_fraction
    if null_fraction > 0:
        warnings.append("Measurement table contains missing values.")

    if not intensity_cols:
        warnings.append("Measurement table has no numeric intensity columns.")
    else:
        zero_cols = []
        for col in intensity_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) and bool(np.allclose(finite, 0.0)):
                zero_cols.append(col)
        metrics["all_zero_intensity_columns"] = zero_cols
        if zero_cols:
            warnings.append("One or more intensity columns are all zero.")

    if not size_cols:
        warnings.append("Measurement table has no area or volume column.")
    else:
        broad_cols: list[str] = []
        negative_cols: list[str] = []
        for col in size_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            if bool((finite < 0).any()):
                negative_cols.append(col)
            positive = finite[finite > 0]
            if len(positive) >= 3:
                p5 = float(np.percentile(positive, 5))
                p95 = float(np.percentile(positive, 95))
                if p5 > 0 and p95 / p5 > 50:
                    broad_cols.append(col)
        metrics["broad_size_distribution_columns"] = broad_cols
        metrics["negative_size_columns"] = negative_cols
        if negative_cols:
            warnings.append("Area or volume columns contain negative values.")
            metrics["failed"] = True
        if broad_cols:
            warnings.append("Object-size distribution is very broad.")

    return _record(table_name, warnings, metrics)


@tool(
    description="Compute QC metrics for a long-format timecourse table. Checks time "
    "and label coverage, missing ROI/time pairs, and flat intensity traces. Stores a "
    "QC record.",
    phase="6",
    worker=True,
)
def compute_timecourse_qc(table_name: str) -> dict[str, Any]:
    df = state.get_table(table_name)
    warnings: list[str] = []
    intensity_cols = _intensity_columns(df)
    time_col = _time_column(df)
    metrics: dict[str, Any] = {
        "kind": "timecourse",
        "table_name": table_name,
        "n_rows": int(len(df)),
        "intensity_columns": intensity_cols,
        "time_column": time_col,
        "failed": False,
    }

    if df.empty:
        warnings.append("Timecourse table is empty.")
        metrics["failed"] = True
        return _record(table_name, warnings, metrics)
    if "label" not in df.columns:
        warnings.append("Timecourse table has no label column.")
        metrics["failed"] = True
    if time_col is None:
        warnings.append("Timecourse table has no time column.")
        metrics["failed"] = True
    if not intensity_cols:
        warnings.append("Timecourse table has no numeric intensity columns.")
        metrics["failed"] = True
    if metrics["failed"]:
        return _record(table_name, warnings, metrics)

    labels = sorted(pd.unique(df["label"].dropna()).tolist())
    times = sorted(pd.unique(df[time_col].dropna()).tolist())
    metrics["n_rois"] = int(len(labels))
    metrics["n_timepoints"] = int(len(times))

    pairs = set(zip(df["label"], df[time_col], strict=False))
    expected_pairs = {(label, time) for label in labels for time in times}
    missing_pairs = expected_pairs - pairs
    metrics["missing_roi_time_pairs"] = int(len(missing_pairs))
    metrics["missing_roi_time_fraction"] = float(
        len(missing_pairs) / max(1, len(expected_pairs))
    )
    if missing_pairs:
        warnings.append("Some ROI/timepoint measurements are missing.")

    primary = intensity_cols[0]
    flat = 0
    point_counts: list[int] = []
    for label, group in df.groupby("label"):
        values = pd.to_numeric(group.sort_values(time_col)[primary], errors="coerce")
        values = values[np.isfinite(values)]
        point_counts.append(int(len(values)))
        if len(values) >= 2 and float(np.nanmax(values) - np.nanmin(values)) <= 1e-9:
            flat += 1
    metrics["points_per_roi_min"] = int(min(point_counts) if point_counts else 0)
    metrics["points_per_roi_max"] = int(max(point_counts) if point_counts else 0)
    metrics["flat_trace_fraction"] = float(flat / max(1, len(labels)))
    if metrics["flat_trace_fraction"] > 0.5:
        warnings.append("Most ROI traces are flat; confirm the time axis and ROI mapping.")

    return _record(table_name, warnings, metrics)


@tool(
    description="Create an additive outline image layer from a Labels layer for visual "
    "review of segmentation boundaries.",
    phase="6",
    worker=True,
)
def create_label_outline(labels_layer: str, name: str | None = None) -> dict[str, Any]:
    from skimage.segmentation import find_boundaries

    labels_snapshot = call_on_main(snapshot_layer, labels_layer)
    labels = _materialize(labels_snapshot.data).astype(np.int64, copy=False)
    outline = find_boundaries(labels, mode="outer").astype(np.uint8)
    layer = call_on_main(
        add_image_from_worker,
        outline,
        name=name or f"{labels_layer}_outline",
        scale=tuple(labels_snapshot.scale),
        metadata={"source_layer": labels_layer, "op": "label_outline"},
        colormap="red",
        blending="additive",
    )
    return {
        "new_layer": layer.name,
        "source_layer": labels_layer,
        "shape": tuple(int(s) for s in outline.shape),
        "n_outline_pixels": int(np.count_nonzero(outline)),
    }


@tool(
    description="Jump napari to a measured object by label id. Uses centroid columns "
    "from a measurement table and selects the matching label in the linked Labels "
    "layer when available.",
    phase="6",
)
def jump_to_object(table_name: str, label: int) -> dict[str, Any]:
    df = state.get_table(table_name)
    if "label" not in df.columns:
        raise ValueError(f"Table {table_name!r} has no label column.")
    matches = df[df["label"].astype(int) == int(label)]
    if matches.empty:
        raise ValueError(f"Label {label!r} not found in table {table_name!r}.")

    entry = state.get_table_entry(table_name)
    labels_layer = entry.spec.get("labels_layer")
    viewer = state.get_viewer()
    selected = False
    if labels_layer and labels_layer in [L.name for L in viewer.layers]:
        layer = viewer.layers[labels_layer]
        try:
            layer.selected_label = int(label)
            layer.show_selected_label = True
            selected = True
        except Exception:
            selected = False

    row = matches.iloc[0]
    center = _centroid_from_row(row, str(labels_layer) if labels_layer else None)
    if center is not None:
        try:
            viewer.camera.center = tuple(float(c) for c in center)
        except Exception:
            pass

    return {
        "table_name": table_name,
        "label": int(label),
        "labels_layer": labels_layer,
        "selected": selected,
        "center": center,
    }


@tool(
    description="Mark a QC record as pass, warning, fail, or not_checked with optional "
    "human review notes. Existing QC metrics and warnings are preserved.",
    phase="6",
)
def mark_qc_status(
    source: str,
    status: str,
    notes: str | None = None,
) -> dict[str, Any]:
    if status not in _QC_STATUSES:
        raise ValueError("status must be pass, warning, fail, or not_checked")
    try:
        existing = state.get_qc_record(source)
        warnings = list(existing.warnings)
        metrics = dict(existing.metrics)
    except KeyError:
        warnings = []
        metrics = {}
    state.put_qc_record(
        source=source,
        status=status,  # type: ignore[arg-type]
        warnings=warnings,
        metrics=metrics,
        reviewed_by_user=True,
        notes=notes,
    )
    return {
        "source": source,
        "status": status,
        "reviewed_by_user": True,
        "notes": notes,
    }
