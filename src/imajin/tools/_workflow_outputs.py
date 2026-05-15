from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.analysis.workflow import build_sample_summary
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
    from imajin.agent.state import list_channel_annotations

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
) -> tuple[Path, bool, dict[str, str | None], list[str]]:
    from imajin.results import create_result_bundle, slugify_result_name
    from imajin.result_bundles import (
        active_context_bundle,
        current_sample_slug,
        finalize_bundle_metadata,
        populate_sample_outputs,
        write_combined_csv,
    )

    warnings: list[str] = []
    sample_slug = current_sample_slug() or slugify_result_name(target_layer)
    # Use only the context-var bundle (set by the batch runner via with_active_bundle)
    # so that standalone calls don't accidentally inherit an ad-hoc process bundle
    # that a prior tool call may have created.
    parent = active_context_bundle()
    own_bundle = parent is None
    anchor: Path | None = None
    if own_bundle:
        from imajin.anchor import resolve_session_anchor

        file_path = None
        try:
            snap = snapshot_layer(target_layer)
            md = snap.metadata if isinstance(snap.metadata, dict) else {}
            file_path = md.get("path") or md.get("source_path")
        except Exception:
            file_path = None
        anchor = resolve_session_anchor(extra_paths=[file_path] if file_path else None)

        bundle_path = create_result_bundle(
            name=f"{target_layer}__{bundle_suffix}",
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
        warnings.append(
            f"bundle outputs could not be written: {type(exc).__name__}: {exc}"
        )
    if own_bundle:
        summary = build_sample_summary(
            sample_name=target_layer,
            status="complete",
            outputs=bundle_outputs,
            source_layer=target_layer,
            **dict(sample_summary or {}),
        )
        try:
            write_combined_csv(bundle_path, table_names)
            finalize_bundle_metadata(
                bundle_path,
                samples=[summary],
                status="complete",
                extra={"run_context_extras": _single_bundle_run_context_extras(anchor)},
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"bundle could not be finalized: {type(exc).__name__}: {exc}")

    return bundle_path, own_bundle, bundle_outputs, warnings
