from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from imajin import result_bundles as _bundle_io
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_table, get_table_entry
from imajin.paths import normalize_user_path
from imajin.result_bundles import (
    bundle_output_path,
    register_output,
    register_table_spec,
)
from imajin.results import (
    create_result_bundle,
    slugify_result_name,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool

current_bundle = _bundle_io.current_bundle
current_sample_slug = _bundle_io.current_sample_slug
finalize_bundle_metadata = _bundle_io.finalize_bundle_metadata
populate_sample_outputs = _bundle_io.populate_sample_outputs
with_active_bundle = _bundle_io.with_active_bundle
with_active_sample_slug = _bundle_io.with_active_sample_slug
write_combined_csv = _bundle_io.write_combined_csv
_label_output_dtype = _bundle_io.label_output_dtype
_materialize = _bundle_io.materialize_result_array
_write_label_tiff = _bundle_io.write_label_tiff


def _resolve_output_path(
    path: str | None,
    *,
    category: str,
    filename: str,
) -> Path:
    if path:
        return normalize_user_path(path).resolve()
    return bundle_output_path(category, filename)


def _source_paths_for_layers(layer_names: list[str] | None) -> list[str]:
    paths: list[str] = []
    for layer_name in layer_names or []:
        try:
            snap = call_on_main(snapshot_layer, layer_name)
        except Exception:
            continue
        md = snap.metadata if isinstance(snap.metadata, dict) else {}
        raw = md.get("source_path") or md.get("path")
        if raw:
            paths.append(str(raw))
            continue
        source_layer = md.get("source_layer")
        if not source_layer:
            continue
        try:
            source_snap = call_on_main(snapshot_layer, str(source_layer))
        except Exception:
            continue
        source_md = (
            source_snap.metadata if isinstance(source_snap.metadata, dict) else {}
        )
        source_path = source_md.get("source_path") or source_md.get("path")
        if source_path:
            paths.append(str(source_path))
    return list(dict.fromkeys(paths))


def _anchor_for_layers(layer_names: list[str] | None) -> Path | None:
    from imajin.anchor import resolve_anchor_folder, resolve_session_anchor

    source_paths = _source_paths_for_layers(layer_names)
    if source_paths:
        return resolve_anchor_folder(source_paths)
    return resolve_session_anchor()


@tool(
    description="Save a Labels layer to disk as TIFF. If path is omitted, saves to "
    "the standard Imajin results directory. Use this for persistent masks/ROIs.",
    phase="4",
)
def save_labels(
    labels_layer: str,
    path: str | None = None,
) -> dict[str, Any]:
    layer = call_on_main(snapshot_layer, labels_layer)
    data = _materialize(layer.data)
    out_dtype = _label_output_dtype(data)
    labels = data.astype(out_dtype, copy=False)

    out = _resolve_output_path(
        path,
        category="labels",
        filename=f"{slugify_result_name(labels_layer)}.tif",
    )
    _write_label_tiff(out, labels)
    register_output(
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
    bundle = create_result_bundle(
        name,
        kind="single",
        metadata=metadata,
        root=_anchor_for_layers(labels_layers),
    )
    outputs: dict[str, list[str]] = {
        "labels": [],
        "tables": [],
        "qc": [],
        "figures": [],
    }

    with with_active_bundle(bundle):
        for labels_layer in labels_layers or []:
            layer = call_on_main(snapshot_layer, labels_layer)
            data = _materialize(layer.data)
            labels = data.astype(_label_output_dtype(data), copy=False)
            out = bundle / "labels" / "cells" / f"{slugify_result_name(labels_layer)}.tif"
            _write_label_tiff(out, labels)
            outputs["labels"].append(str(out))
            register_output(
                "labels_tiff",
                out,
                {
                    "labels_layer": labels_layer,
                    "shape": tuple(int(s) for s in labels.shape),
                    "dtype": str(labels.dtype),
                },
            )

        for table_name in table_names or []:
            df = get_table(table_name)
            out = bundle / "tables" / f"{slugify_result_name(table_name)}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            outputs["tables"].append(str(out))
            register_table_spec(table_name, get_table_entry(table_name).spec)
            register_output("table_csv", out, {"table_name": table_name})

        for raw in qc_png_paths or []:
            if not raw:
                continue
            src = normalize_user_path(raw).resolve()
            if not src.exists():
                continue
            dst = bundle / "qc" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            outputs["qc"].append(str(dst))
            register_output("qc_png", dst, {"source": str(src)})

        for raw in figures or []:
            if not raw:
                continue
            src = normalize_user_path(raw).resolve()
            if not src.exists():
                continue
            dst = bundle / "figures" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            outputs["figures"].append(str(dst))
            register_output("figure", dst, {"source": str(src)})

        register_output(
            "result_bundle",
            bundle / "metadata.json",
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
