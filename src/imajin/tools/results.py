from __future__ import annotations

import contextlib
import contextvars
import shutil
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_table, get_table_entry
from imajin.paths import normalize_user_path
from imajin.results import (
    create_result_bundle,
    read_bundle_metadata,
    record_result,
    slugify_result_name,
    unique_result_path,
    write_bundle_metadata,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


_active_bundle: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "imajin_active_bundle", default=None
)
_active_sample_slug: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "imajin_active_sample_slug", default=None
)


def current_bundle() -> Path | None:
    """Return the bundle directory currently being populated, or None.

    Set by `with_active_bundle` (used by run_recipe_on_samples to forward
    the parent bundle to per-sample analyze_target_cells calls).
    """
    return _active_bundle.get()


def current_sample_slug() -> str | None:
    """Return the slug of the sample currently being processed, or None.

    Set by `with_active_sample_slug` so per-call writers can name files by
    sample name in a batch instead of by layer name.
    """
    return _active_sample_slug.get()


@contextlib.contextmanager
def with_active_bundle(path: Path | str) -> Iterator[Path]:
    """Mark `path` as the current bundle for the duration of the with-block.

    Note: Python `contextvars` propagate to `threading.Thread` children but NOT
    to Qt thread-pool workers (e.g. napari's `@thread_worker`). If a caller
    inside this block dispatches work via such a pool, that work will see
    `current_bundle() is None`. Callers in such situations must capture the
    path explicitly or run the dispatched callable inside a copied context
    (`contextvars.copy_context().run(...)`).
    """
    p = Path(path)
    token = _active_bundle.set(p)
    try:
        yield p
    finally:
        _active_bundle.reset(token)


@contextlib.contextmanager
def with_active_sample_slug(slug: str | None) -> Iterator[str | None]:
    """Mark a per-sample slug used by per-call bundle writers.

    Same propagation caveats as `with_active_bundle` apply.
    """
    token = _active_sample_slug.set(slug)
    try:
        yield slug
    finally:
        _active_sample_slug.reset(token)


def _materialize(arr: Any) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _label_output_dtype(data: np.ndarray) -> np.dtype:
    max_label = int(np.nanmax(data)) if data.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _resolve_output_path(
    path: str | None,
    *,
    category: str,
    filename: str,
    bundle: Path | None = None,
) -> Path:
    if path:
        return normalize_user_path(path).resolve()
    if bundle is not None:
        return bundle / category / filename
    return unique_result_path(category, filename)


@tool(
    description="Save a Labels layer to disk as TIFF. If path is omitted, saves to "
    "the standard Imajin results directory (project reports if a project is open; "
    "otherwise the user's Imajin results folder). Use this for persistent masks/ROIs.",
    phase="4",
)
def save_labels(
    labels_layer: str,
    path: str | None = None,
) -> dict[str, Any]:
    import tifffile

    layer = call_on_main(snapshot_layer, labels_layer)
    data = _materialize(layer.data)
    out_dtype = _label_output_dtype(data)
    labels = data.astype(out_dtype, copy=False)

    out = _resolve_output_path(
        path,
        category="labels",
        filename=f"{slugify_result_name(labels_layer)}.tif",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out, labels)
    record_result(
        "labels_tiff",
        out,
        {
            "labels_layer": labels_layer,
            "shape": tuple(int(s) for s in labels.shape),
            "dtype": str(labels.dtype),
        },
    )
    return {
        "path": str(out),
        "labels_layer": labels_layer,
        "shape": tuple(int(s) for s in labels.shape),
        "dtype": str(labels.dtype),
        "n_labels": int(labels.max()) if labels.size else 0,
    }


@tool(
    description="Save a result bundle containing labels TIFFs, measurement table CSVs, "
    "QC PNGs, and metadata.json. Use after segmentation/measurement so all generated "
    "outputs are in one folder.",
    phase="4",
)
def save_result_bundle(
    name: str,
    labels_layers: list[str] | None = None,
    table_names: list[str] | None = None,
    qc_png_paths: list[str] | None = None,
    figures: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import tifffile

    bundle = create_result_bundle(name, kind="single", metadata=metadata)
    outputs: dict[str, list[str]] = {
        "labels": [],
        "tables": [],
        "qc": [],
        "figures": [],
    }

    for labels_layer in labels_layers or []:
        layer = call_on_main(snapshot_layer, labels_layer)
        data = _materialize(layer.data)
        labels = data.astype(_label_output_dtype(data), copy=False)
        out = bundle / "labels" / "cells" / f"{slugify_result_name(labels_layer)}.tif"
        tifffile.imwrite(out, labels)
        outputs["labels"].append(str(out))

    for table_name in table_names or []:
        df = get_table(table_name)
        out = bundle / "tables" / f"{slugify_result_name(table_name)}.csv"
        df.to_csv(out, index=False)
        outputs["tables"].append(str(out))

        spec_path = bundle / "tables" / f"{slugify_result_name(table_name)}.spec.json"
        spec_path.write_text(
            json.dumps(
                get_table_entry(table_name).spec,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    for raw in qc_png_paths or []:
        if not raw:
            continue
        src = normalize_user_path(raw).resolve()
        if not src.exists():
            continue
        dst = bundle / "qc" / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        outputs["qc"].append(str(dst))

    for raw in figures or []:
        if not raw:
            continue
        src = normalize_user_path(raw).resolve()
        if not src.exists():
            continue
        dst = bundle / "figures" / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        outputs["figures"].append(str(dst))

    bundle_meta = read_bundle_metadata(bundle)
    bundle_meta["outputs"] = outputs
    write_bundle_metadata(bundle, bundle_meta)
    record_result(
        "result_bundle",
        bundle,
        {
            "name": name,
            "n_labels": len(outputs["labels"]),
            "n_tables": len(outputs["tables"]),
            "n_qc": len(outputs["qc"]),
            "n_figures": len(outputs["figures"]),
        },
    )

    return {
        "bundle_path": str(bundle),
        "metadata_path": str(bundle / "metadata.json"),
        "outputs": outputs,
    }


def _write_label_layer(
    bundle: Path, tier: str, sample_slug: str, layer_name: str
) -> str:
    """Snapshot a label layer and write it to bundle/labels/<tier>/<slug>.tif.

    Returns the path relative to `bundle`.
    Raises ValueError on filename collision within the same bundle.
    """
    import tifffile

    layer = call_on_main(snapshot_layer, layer_name)
    data = _materialize(layer.data)
    labels = data.astype(_label_output_dtype(data), copy=False)
    rel = Path("labels") / tier / f"{sample_slug}.tif"
    out = bundle / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(
            f"{rel} already exists in bundle {bundle.name}; "
            "sample_slug collision suspected"
        )
    tifffile.imwrite(out, labels)
    return rel.as_posix()


def _copy_qc_png(bundle: Path, qc_png: str, sample_slug: str) -> str | None:
    """Copy a QC PNG into bundle/qc/<slug>.png. Returns path relative to bundle."""
    src = normalize_user_path(qc_png).resolve()
    if not src.exists():
        return None
    rel = Path("qc") / f"{sample_slug}.png"
    dst = bundle / rel
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return rel.as_posix()


def populate_sample_outputs(
    bundle: Path,
    sample_slug: str,
    *,
    labels_cells: str | None = None,
    labels_domain: str | None = None,
    qc_png: str | None = None,
) -> dict[str, str | None]:
    """Write per-sample outputs into a bundle, returning relative output paths."""
    out: dict[str, str | None] = {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }
    if labels_cells:
        out["labels_cells"] = _write_label_layer(
            bundle, "cells", sample_slug, labels_cells
        )
    if labels_domain:
        out["labels_domain"] = _write_label_layer(
            bundle, "domain", sample_slug, labels_domain
        )
    if qc_png:
        out["qc_png"] = _copy_qc_png(bundle, qc_png, sample_slug)
    return out


def write_combined_csv(bundle: Path, table_names: list[str]) -> Path:
    """Concat the given measurement tables and write to bundle/tables/combined.csv."""
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for name in table_names:
        try:
            frame = get_table(name)
        except KeyError:
            continue
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    out = bundle / "tables" / "combined.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    combined.to_csv(out, index=False)
    return out


def finalize_bundle_metadata(
    bundle: Path,
    *,
    samples: list[dict[str, Any]],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Update bundle/metadata.json with the final samples index and status."""
    meta = read_bundle_metadata(bundle)
    meta["status"] = status
    meta["samples"] = list(samples)
    meta["n_samples"] = len(samples)
    meta["n_complete"] = sum(1 for s in samples if s.get("status") == "complete")
    meta["n_failed"] = sum(1 for s in samples if s.get("status") == "failed")
    meta["tables"] = {"combined": "tables/combined.csv"}
    if extra:
        meta.update(extra)
    write_bundle_metadata(bundle, meta)
