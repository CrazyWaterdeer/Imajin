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
promote_to_process_bundle = _bundle_io.promote_to_process_bundle
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


def _registered_identity(bundle: Path, rel_path: str, key: str) -> str | None:
    """The identity value recorded for an existing output at ``rel_path``."""
    try:
        recorded = _bundle_io.read_bundle_metadata(bundle)
    except Exception:  # noqa: BLE001 - a bad read must not block writing
        return None
    for entry in recorded.get("outputs") or []:
        if not isinstance(entry, dict) or entry.get("path") != rel_path:
            continue
        metadata = entry.get("metadata")
        if isinstance(metadata, dict) and metadata.get(key) is not None:
            return str(metadata[key])
    return None


def _registered_path_for(
    bundle: Path,
    kind: str,
    key: str,
    value: str,
) -> Path | None:
    """Where this bundle already keeps the ``kind`` output for ``key == value``.

    Matching on ``kind`` is load-bearing, not defensive: a segmentation QC PNG
    records the same ``labels_layer`` as the label TIFF it illustrates, so a
    key-only lookup finds the PNG and the caller writes a TIFF over it.
    """
    try:
        recorded = _bundle_io.read_bundle_metadata(bundle)
    except Exception:  # noqa: BLE001 - a bad read just means "not registered"
        return None
    for entry in recorded.get("outputs") or []:
        if not isinstance(entry, dict) or entry.get("kind") != kind:
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict) or str(metadata.get(key)) != value:
            continue
        rel = entry.get("path")
        if rel:
            return bundle / str(rel)
    return None


def _non_clobbering_path(
    bundle: Path,
    category: str,
    filename: str,
    *,
    identity_key: str,
    identity_value: str,
) -> Path:
    """Resolve <bundle>/<category>/<filename>, never overwriting another file's output.

    Re-saving the SAME source (same labels layer, same QC image) overwrites in
    place so a repeated save stays idempotent. A different source landing on a
    taken name gets `_2`, `_3`, ... instead of silently replacing it — which is
    what happened when several files shared a bundle under one layer-derived
    name and six of seven label TIFFs were lost.
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = bundle / category / filename
    index = 1
    while candidate.exists():
        rel = candidate.relative_to(bundle).as_posix()
        prior = _registered_identity(bundle, rel, identity_key)
        if prior is None or prior == identity_value:
            break  # same source (or unattributed) — overwrite in place
        index += 1
        candidate = bundle / category / f"{stem}_{index}{suffix}"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


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
    description="Collect outputs (labels TIFFs, measurement CSVs, QC PNGs, figures, "
    "metadata.json) into the active analysis bundle. By default this APPENDS to the "
    "bundle opened by start_analysis — or the one an earlier save/output created in this "
    "task — so a sequential multi-file workflow (e.g. per-file ROI analysis) lands in ONE "
    "folder. A new folder is created only when no bundle is active; pass new_bundle=True to "
    "force a separate folder for a genuinely independent result.",
    phase="4",
)
def save_result_bundle(
    name: str,
    labels_layers: list[str] | None = None,
    table_names: list[str] | None = None,
    qc_png_paths: list[str] | None = None,
    figures: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    new_bundle: bool = False,
) -> dict[str, Any]:
    # Reuse the active bundle (start_analysis, a batch context, or an earlier output
    # in this task) so per-file results accumulate in one folder. Only mint a new
    # bundle when none is active or the caller forces it, and promote a fresh one to
    # the process slot so subsequent tool calls in the same task share it too.
    bundle = None if new_bundle else current_bundle()
    reused = bundle is not None
    if reused and metadata:
        # Don't drop caller metadata just because we're appending.
        seed = _bundle_io.read_bundle_metadata(bundle)
        recipe_params = dict(seed.get("recipe_params") or {})
        recipe_params.update(metadata)
        seed["recipe_params"] = recipe_params
        _bundle_io.write_bundle_metadata(bundle, seed)
    if bundle is None:
        bundle = create_result_bundle(
            name,
            kind="single",
            metadata=metadata,
            root=_anchor_for_layers(labels_layers),
        )
        promote_to_process_bundle(bundle)
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
            # If the analysis already wrote this layer into the bundle (under
            # its per-file name), update that file instead of laying down a
            # second full-resolution copy under the layer-derived name.
            out = _registered_path_for(
                bundle, "labels_tiff", "labels_layer", labels_layer
            ) or _non_clobbering_path(
                bundle,
                "labels/cells",
                f"{slugify_result_name(labels_layer)}.tif",
                identity_key="labels_layer",
                identity_value=labels_layer,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
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
            dst = _non_clobbering_path(
                bundle,
                "qc",
                src.name,
                identity_key="source",
                identity_value=str(src),
            )
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
            dst = _non_clobbering_path(
                bundle,
                "figures",
                src.name,
                identity_key="source",
                identity_value=str(src),
            )
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
        "reused": reused,
        # When appending, `name` labels the outputs but does NOT rename the
        # folder — say so, instead of letting the caller believe its per-sample
        # name was applied. In the reported session the agent passed good names
        # (mF_rectum_1 ... vF_rectum_2) that never reached any folder.
        "name_applied": not reused,
        "bundle_name": (
            _bundle_io.read_bundle_metadata(bundle).get("run_context") or {}
        ).get("name"),
    }
