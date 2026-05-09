from __future__ import annotations

import shutil
import json
from pathlib import Path
from typing import Any

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
