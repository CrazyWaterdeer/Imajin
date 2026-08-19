from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.workflow import build_sample_summary
from imajin.session import list_channel_annotations
from imajin.tools.napari_ops import snapshot_layer


def _empty_bundle_outputs() -> dict[str, str | None]:
    return {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }


def _bundle_qc_png_path(
    bundle_path: Path | None,
    bundle_outputs: dict[str, str | None],
    fallback: str | None,
) -> str | None:
    rel = bundle_outputs.get("qc_png")
    if bundle_path is not None and rel:
        return str((bundle_path / rel).resolve())
    return fallback


def _single_bundle_run_context_extras(anchor: Path | None) -> dict[str, Any]:

    channel_roles: dict[str, str] = {}
    for entry in list_channel_annotations():
        layer_name = entry.get("layer_name")
        role = entry.get("role")
        if layer_name and role:
            channel_roles[str(layer_name)] = str(role)

    return {
        "folder_set": [str(anchor)] if anchor is not None else [],
        "channel_roles": channel_roles,
        "scope_filters": [],
    }


def _reusable_session_bundle() -> Path | None:
    """The open session bundle to append this file's outputs to, if there is one.

    Reuses the process-slot bundle only when its own metadata says it is a
    deliberately-opened, still-open analysis:

    * ``kind != "adhoc"`` — a bundle lazily minted by the first stray output is
      not a session; adopting one lets a QC PNG decide where an analysis lands.
    * ``status == "in_progress"`` — never append to a finalized bundle. A closed
      folder that keeps accepting writes is how each file's QC ended up filed
      under the previous file.

    Discriminates on the metadata ``kind`` field, not the ``_adhoc`` folder-name
    suffix, which misfires on the ``_adhoc_2``/``_adhoc_3`` dedup variants.
    """
    from imajin.result_bundles import current_bundle, read_bundle_metadata_normalized

    bundle = current_bundle()
    if bundle is None:
        return None
    try:
        run_context = (
            read_bundle_metadata_normalized(bundle).get("run_context") or {}
        )
    except Exception:  # noqa: BLE001 - an unreadable bundle is not reusable
        return None
    if str(run_context.get("kind") or "") == "adhoc":
        return None
    if str(run_context.get("status") or "") != "in_progress":
        return None
    return bundle


def _sample_identity(
    *,
    bundle: Path | None,
    source_file: str | None,
    target_layer: str,
    anchor: Path | None,
) -> tuple[str, str]:
    """Return ``(sample_name, sample_slug)`` for one per-file analysis.

    Identity comes from the SOURCE FILE, not the layer. Channel/layer names come
    from instrument metadata (``Ch2-T1``) and repeat across every file in a
    folder, so a layer-derived slug makes all of a session's files collide on one
    name — which silently overwrote six of seven label TIFFs once the files
    started sharing a bundle.

    Falls back to the layer name for in-memory layers that have no source file.
    If the stem is already taken in this bundle by a *different* file, the
    collision-resistant ``<stem>_<8hex>`` slug is used instead so both survive.
    """
    from imajin.results import slugify_result_name

    if not source_file:
        slug = slugify_result_name(target_layer)
        return target_layer, slug

    from imajin.analysis.resume import rel_key, sample_slug_for

    key = rel_key(source_file, anchor) if anchor is not None else str(source_file)
    stem = Path(key).stem or target_layer
    slug = slugify_result_name(stem)

    if bundle is not None:
        from imajin.result_bundles import read_bundle_metadata_normalized

        try:
            recorded = read_bundle_metadata_normalized(bundle)
            samples = (recorded.get("run_context") or {}).get("samples") or []
        except Exception:  # noqa: BLE001 - identity must not fail on a bad read
            samples = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if slugify_result_name(str(sample.get("sample_name") or "")) != slug:
                continue
            prior = sample.get("source_file")
            if prior and str(prior) != str(source_file):
                return stem, sample_slug_for(key)
    return stem, slug


def _write_analysis_bundle_outputs(
    *,
    target_layer: str,
    target_source: str,
    segmentation_method: str,
    analysis_dim: str,
    tier: str,
    bundle_suffix: str,
    table_names: list[str],
    labels_cells: str,
    labels_domain: str | None = None,
    qc_png: str | None = None,
    sample_summary: dict[str, Any] | None = None,
    source_file: str | None = None,
) -> tuple[Path, bool, bool, dict[str, str | None], list[str]]:
    from imajin.results import create_result_bundle, slugify_result_name
    from imajin.result_bundles import (
        active_context_bundle,
        current_sample_slug,
        finalize_bundle_metadata,
        populate_sample_outputs,
        write_combined_csv,
    )

    warnings: list[str] = []
    from imajin.anchor import resolve_session_anchor

    file_path = source_file
    if not file_path:
        try:
            snap = call_on_main(snapshot_layer, target_layer)
            md = snap.metadata if isinstance(snap.metadata, dict) else {}
            file_path = md.get("path") or md.get("source_path")
        except Exception:  # noqa: BLE001 - identity falls back to the layer name
            file_path = None
    anchor = resolve_session_anchor(extra_paths=[file_path] if file_path else None)

    # Which bundle do this file's outputs belong to?
    #
    # The contextvar (set by the batch runner via with_active_bundle) wins. Then
    # a bundle a human or agent DELIBERATELY opened — start_analysis, or an
    # earlier per-file analysis in this same session — so a sequential one-file-
    # at-a-time run collects in one folder instead of minting a folder per file.
    #
    # Provenance is read off the bundle, not inferred from which slot it sits in.
    # Encoding it as "contextvar = deliberate, process slot = ad-hoc" was the
    # original defect: it correctly refused stray ad-hoc bundles but threw away
    # every deliberately-opened one too, because nothing an agent can call sets
    # the contextvar.
    ctx_bundle = active_context_bundle()
    # Only look for a session bundle when the batch runner is NOT driving; the
    # batch runner owns its parent bundle's sample bookkeeping end to end.
    session_bundle = None if ctx_bundle is not None else _reusable_session_bundle()
    parent = ctx_bundle if ctx_bundle is not None else session_bundle
    own_bundle = parent is None
    batch_managed = ctx_bundle is not None
    sample_name, sample_slug = _sample_identity(
        bundle=parent,
        source_file=file_path,
        target_layer=target_layer,
        anchor=anchor,
    )
    batch_slug = current_sample_slug()
    if batch_slug:
        # The batch runner owns per-sample identity; don't second-guess it.
        sample_slug = batch_slug
    if own_bundle:
        bundle_name = (
            f"{slugify_result_name(sample_name)}__{bundle_suffix}"
            if file_path
            else f"{target_layer}__{bundle_suffix}"
        )
        bundle_path = create_result_bundle(
            name=bundle_name,
            kind="single",
            tier=tier,
            metadata={
                "recipe": None,
                "target_channel": target_layer,
                "target_source": target_source,
                "segmentation_method": segmentation_method,
                "analysis_dim": analysis_dim,
            },
            root=anchor,
        )
        from imajin.result_bundles import promote_to_process_bundle
        promote_to_process_bundle(bundle_path)
    else:
        bundle_path = parent

    bundle_outputs = _empty_bundle_outputs()
    try:
        bundle_outputs = populate_sample_outputs(
            bundle_path,
            sample_slug=sample_slug,
            labels_cells=labels_cells,
            labels_domain=labels_domain,
            qc_png=qc_png,
        )
    except Exception as exc:  # noqa: BLE001
        # Loud, not decorative: the sample's labels/QC are NOT in the bundle, so
        # the summary below will record nulls for them. Silently degrading here
        # is how a bundle ends up claiming a sample it has no outputs for.
        warnings.append(
            f"sample {sample_name!r} outputs were NOT written into the bundle "
            f"({type(exc).__name__}: {exc}) — its labels/QC are missing"
        )
    # Record this sample unless the batch runner is doing it. When appending to
    # a session bundle the run is NOT over, so the status stays in_progress —
    # "finalized" and "still the default write target" must never both be true.
    summary = build_sample_summary(
        sample_name=sample_name,
        status="complete",
        outputs=bundle_outputs,
        source_layer=target_layer,
        source_file=str(file_path) if file_path else None,
        **dict(sample_summary or {}),
    )
    try:
        if own_bundle:
            write_combined_csv(bundle_path, table_names)
        if not batch_managed:
            finalize_bundle_metadata(
                bundle_path,
                samples=[summary],
                status="complete" if own_bundle else "in_progress",
                extra={
                    "run_context_extras": _single_bundle_run_context_extras(anchor)
                }
                if own_bundle
                else {},
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"sample {sample_name!r} could not be recorded in the bundle: "
            f"{type(exc).__name__}: {exc}"
        )

    return bundle_path, own_bundle, batch_managed, bundle_outputs, warnings
