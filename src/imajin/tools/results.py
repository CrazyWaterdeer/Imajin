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
    place so a repeated save stays idempotent. Anything else landing on a taken
    name gets `_2`, `_3`, ... instead of silently replacing it — which is what
    happened when several files shared a bundle under one layer-derived name and
    six of seven label TIFFs were lost.

    An existing file that this bundle's index does not attribute to anyone is
    treated as somebody else's, not as free space: a pre-existing bundle written
    by an older version registers nothing, and overwriting it would be the same
    data loss by another route.
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = bundle / category / filename
    index = 1
    while candidate.exists():
        rel = candidate.relative_to(bundle).as_posix()
        if _registered_identity(bundle, rel, identity_key) == identity_value:
            break  # our own earlier output — overwrite in place
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


def _sample_slug_for_layer(layer_name: str) -> str | None:
    """Per-file identity for a layer, derived from the file behind it.

    Mirrors the rule the analysis path uses (see
    ``imajin.tools._workflow_outputs._sample_identity``) so both writers agree
    on one name per source file. Returns ``None`` for an in-memory layer with no
    file behind it, and the caller falls back to the layer name.
    """
    from imajin.analysis.resume import rel_key
    from imajin.anchor import resolve_session_anchor

    sources = _source_paths_for_layers([layer_name])
    if not sources:
        return None
    source = sources[0]
    anchor = resolve_session_anchor(extra_paths=[source])
    key = rel_key(source, anchor) if anchor is not None else str(source)
    stem = Path(key).stem
    return slugify_result_name(stem) if stem else None


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

    slug = _sample_slug_for_layer(labels_layer)
    if path:
        out = normalize_user_path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Same per-file identity + no-clobber rule as save_result_bundle: an
        # explicit path is the caller's business, a default one must not
        # overwrite another file's labels.
        bundle = _bundle_io.ensure_active_bundle()
        out = _non_clobbering_path(
            bundle,
            "labels",
            f"{slugify_result_name(slug or labels_layer)}.tif",
            identity_key="sample_slug" if slug else "labels_layer",
            identity_value=slug or labels_layer,
        )
    _write_label_tiff(out, labels)
    register_output(
        "labels_tiff",
        out,
        {
            "labels_layer": labels_layer,
            "sample_slug": slug,
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
    "force a separate folder for a genuinely independent result — that writes the one "
    "result aside and leaves the active bundle in place for everything after it.",
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
        if new_bundle and current_bundle() is not None:
            # `new_bundle=True` means "put this ONE result somewhere separate",
            # not "switch the session". Promoting here made every later
            # analysis and save land in the sidecar folder instead of the
            # session the user opened, with nothing reporting the switch.
            pass
        else:
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
            # Identity comes from the SOURCE FILE, exactly as the analysis path
            # derives it. Keying on the layer name here would be the original
            # bug in miniature: every file segments to `Ch2-T1_objects`, so the
            # lookup would return the FIRST file's TIFF and each subsequent save
            # would overwrite it.
            slug = _sample_slug_for_layer(labels_layer)
            out = (
                _registered_path_for(bundle, "labels_tiff", "sample_slug", slug)
                if slug
                else None
            ) or _non_clobbering_path(
                bundle,
                "labels/cells",
                f"{slugify_result_name(slug or labels_layer)}.tif",
                identity_key="sample_slug" if slug else "labels_layer",
                identity_value=slug or labels_layer,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_label_tiff(out, labels)
            outputs["labels"].append(str(out))
            register_output(
                "labels_tiff",
                out,
                {
                    "labels_layer": labels_layer,
                    "sample_slug": slug,
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
