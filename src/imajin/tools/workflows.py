from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.analysis.workflow import (
    check_analysis_memory_budget as _check_analysis_memory_budget,
    decide_3d as _decide_3d,
    derive_size_params as _derive_size_params,  # noqa: F401 - re-exported (used by tests/_workflow_steps)
    normalize_domain_spec as _normalize_domain_spec,  # noqa: F401 - re-exported (used by experiment.py)
    normalize_segmentation_method as _normalize_segmentation_method,
)
from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import (
    AmbiguousChannelError,
    get_layer as _get_layer,
    get_table,
    list_runs as _list_runs,
    put_run as _put_run,
    put_table,
    resolve_target_channel,
)
from imajin.tools.batch_runner import BatchRecipeRunner
from imajin.tools.files import _layer_source_path
from imajin.tools import measure as _measure
from imajin.tools._workflow_outputs import (
    _bundle_qc_png_path,
    _empty_bundle_outputs,
    _write_analysis_bundle_outputs,
    pin_analysis_bundle,
)
from imajin.tools._workflow_steps import (
    _layer_axes,
    _precompute_domain_layer,
    _run_preprocess_step,
    _run_segmentation_step,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


def _run_label(file_key: str) -> str:
    """Short display label for an analysis key (file stem, or the layer name)."""
    if "/" in file_key or "\\" in file_key:
        return Path(file_key).stem
    return file_key


def _latest_complete_run(file_key: str, recipe_key: str) -> dict[str, Any] | None:
    """The stored result of the newest *complete* interactive run for this key."""
    for rec in reversed(_list_runs()):
        if (
            rec.get("file_id") == file_key
            and rec.get("recipe_id") == recipe_key
            and rec.get("status") == "complete"
        ):
            result = (rec.get("summary") or {}).get("result")
            if isinstance(result, dict):
                return dict(result)
    return None


def _record_interactive_run(
    file_key: str, recipe_key: str, status: str, result: dict[str, Any]
) -> None:
    """Record an AnalysisRun for the interactive path so the batch-progress ledger
    and the re-run guard share one source of truth with the batch runner."""
    table = result.get("primary_table_name") or result.get("table_name")
    layers = [
        name
        for name in (
            result.get("cells_layer") or result.get("labels_layer"),
            result.get("domain_layer"),
        )
        if name
    ]
    _put_run(
        sample_id=_run_label(file_key),
        file_id=file_key,
        recipe_id=recipe_key,
        status=status,
        table_names=[table] if table else [],
        layer_names=layers,
        summary={
            "n_cells": result.get("n_cells") or result.get("n_objects"),
            "method": recipe_key,
            "result": result,
        },
    )


def _resume_skip(analysis_file_key: str) -> dict[str, Any] | None:
    """Skip a file a resumed bundle already covers, matched by anchor-relative key.

    Enforced here (not just in the prompt) so re-running a done file during resume is
    impossible by default, and robust across mounts/platforms (WSL↔Windows) where the
    absolute path — and thus the ledger's exact key — would differ.
    """
    from imajin.session import get_resume_scope

    scope = get_resume_scope()
    if not scope:
        return None
    from imajin.analysis.resume import rel_key

    key = rel_key(analysis_file_key, scope["anchor"])
    if key in scope["done_keys"]:
        return {
            "ok": True,
            "already_analysed": True,
            "resumed_skip": True,
            "message": (
                f"{_run_label(analysis_file_key)} is already in the resumed bundle "
                f"({key}); pass rerun=True to recompute."
            ),
        }
    return None


@tool(
    description="High-level target object analysis workflow. Resolves the target "
    "channel, optionally preprocesses it, segments target-positive objects/ROIs, "
    "and measures per-object intensity and size on the same target channel. Pass "
    "target as a layer name, color (green/red/UV/IR), or marker (GFP/DAPI). Pass preprocess="
    "'rolling_ball' / 'auto_contrast' / 'gaussian_denoise' to apply one preprocessing "
    "step before segmentation. Returns labels layer, measurement table, object count, "
    "and QC metrics. Counterstain channels are not auto-selected — annotate them as "
    "'counterstain' first if you only have one target channel. Pass region_mask=<a "
    "boundary Labels layer, e.g. from boundary_mask_from_shapes> to constrain BOTH the "
    "expression domain and the cells to inside a hand-drawn region (target_objects / "
    "auto_3d_cells only); the heavy compute is cropped to that region.",
    phase="2",
    worker=True,
)
def analyze_target_cells(
    target: str | None = None,
    do_3D: bool | None = None,
    diameter: float | None = None,
    preprocess: str | None = None,
    segmentation_method: str = "target_objects",
    segmentation_options: dict[str, Any] | None = None,
    region_mask: str | None = None,
    domain_strategy: str | None = None,
    domain_options: dict[str, Any] | None = None,
    counterstain_layer: str | None = None,
    cell_diameter_um: float | None = None,
    review_mode: str = "auto",
    review_timeout_s: float | None = None,
    rerun: bool = False,
    batch_managed: bool = False,
) -> dict[str, Any]:
    warnings: list[str] = []
    report_progress(stage="resolve_target", message="Resolving target channel.")
    try:
        resolution = resolve_target_channel(target)
    except AmbiguousChannelError as e:
        return {
            "ok": False,
            "error": str(e),
            "candidates": list(e.candidates),
            "stage": "resolve_target",
        }

    target_layer = resolution.layer
    if resolution.source == "inference":
        warnings.append(
            f"single image layer ({target_layer}) was assumed as target — "
            "confirm by annotating the channel."
        )

    # Resolve the analysis identity from the ORIGINAL target (before preprocessing)
    # so the re-run guard short-circuits a duplicate before any heavy work runs.
    method = _normalize_segmentation_method(segmentation_method)
    mode = "two_tier" if domain_strategy is not None else "single"
    orig_snapshot = call_on_main(snapshot_layer, target_layer)
    # None for an in-memory layer with no file behind it; the bundle falls back
    # to the layer name for identity in that case.
    analysis_source_file = _layer_source_path(orig_snapshot)
    analysis_file_key = analysis_source_file or target_layer
    analysis_recipe_key = f"interactive:{method}:{mode}"

    if not batch_managed and not rerun:
        prior_result = _latest_complete_run(analysis_file_key, analysis_recipe_key)
        if prior_result is not None:
            prior_table = (
                prior_result.get("primary_table_name")
                or prior_result.get("table_name")
            )
            return {
                **prior_result,
                "already_analysed": True,
                "message": (
                    f"{_run_label(analysis_file_key)} was already analysed "
                    f"(table {prior_table}); pass rerun=True to recompute."
                ),
            }
        resumed = _resume_skip(analysis_file_key)
        if resumed is not None:
            return resumed

    seg_input_layer, pre_step, pre_record = _run_preprocess_step(
        target_layer,
        preprocess,
    )

    snapshot = call_on_main(snapshot_layer, seg_input_layer)
    axes = _layer_axes(snapshot)
    use_3d = _decide_3d(do_3D, axes, getattr(snapshot.data, "ndim", 2))

    # Pin the destination bundle BEFORE the domain pre-compute and segmentation,
    # both of which write QC PNGs through ensure_active_bundle(). Deciding it
    # afterwards let a single file's outputs straddle two folders.
    analysis_bundle = pin_analysis_bundle(
        target_layer=target_layer,
        target_source=resolution.source,
        segmentation_method=method,
        analysis_dim="3d" if use_3d else "2d",
        tier="two_tier" if domain_strategy is not None else "single_tier",
        bundle_suffix="two_tier" if domain_strategy is not None else "single",
        source_file=analysis_source_file,
    )

    # A hand-drawn ROI (region_mask) constrains BOTH tiers to inside the region.
    if region_mask is not None:
        if method not in {"target_objects", "auto_3d_cells"}:
            return {
                "ok": False,
                "stage": "region_mask",
                "error": f"region_mask only works with segmentation_method "
                f"'target_objects' or 'auto_3d_cells' (got {method!r}); "
                "cellpose_sam / intensity_regions have no boundary support.",
            }
        if "boundary_mask" in (segmentation_options or {}):
            return {
                "ok": False,
                "stage": "region_mask",
                "error": "pass the ROI via region_mask OR "
                "segmentation_options['boundary_mask'], not both.",
            }

    # Tier-1: pre-compute expression domain before Tier-2 segmentation. When a
    # region_mask is given the domain is ROI-constrained, so it is measured only
    # inside the drawn region and Tier-2 (which uses the domain as its boundary) is
    # transitively constrained too.
    pre_computed_domain_layer: str | None = None
    if domain_strategy is not None:
        pre_computed_domain_layer = _precompute_domain_layer(
            target_layer=target_layer,
            snapshot=snapshot,
            domain_options=domain_options,
            counterstain_layer=counterstain_layer,
            cell_diameter_um=cell_diameter_um,
            boundary_mask=region_mask,
        )

    _check_analysis_memory_budget(seg_input_layer, data=snapshot.data, method=method)

    seg_options = dict(segmentation_options or {})
    seg_options.pop("tool", None)
    seg_options.pop("do_3D", None)
    seg_options.pop("diameter", None)
    if pre_computed_domain_layer is not None:
        seg_options.setdefault("boundary_mask", pre_computed_domain_layer)
        seg_options.setdefault("min_snr", 1.6)
        seg_options.setdefault("high_snr", 3.2)
    elif region_mask is not None:
        # Single-tier: constrain Tier-2 directly to the ROI.
        seg_options.setdefault("boundary_mask", region_mask)
    seg_result = _run_segmentation_step(
        method=method,
        seg_input_layer=seg_input_layer,
        seg_options=seg_options,
        use_3d=use_3d,
        diameter=diameter,
    )
    if seg_result.get("empty_mask", False):
        empty_result = {
            "ok": False,
            "stage": "segment",
            "error": f"{method} produced zero objects on the target channel; "
            "no measurements were taken. Try a different channel, a preprocess "
            "step (rolling_ball / auto_contrast / gaussian_denoise), or set a "
            "manual diameter.",
            "target_channel": target_layer,
            "target_source": resolution.source,
            "labels_layer": seg_result["labels_layer"],
            "qc_png_path": seg_result.get("qc_png_path"),
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "preprocess": pre_step,
            "segmentation_method": method,
            "warnings": warnings,
        }
        if not batch_managed:
            # Record a failed run so the ledger shows it; a failed run never blocks a retry.
            _record_interactive_run(
                analysis_file_key, analysis_recipe_key, "failed", empty_result
            )
        return empty_result

    review_record: dict[str, Any] | None = None
    if review_mode == "interactive":
        from imajin.agent.review_checkpoint import request_review_and_wait

        report_progress(
            stage="review",
            message=(
                f"Awaiting interactive review of {seg_result['labels_layer']}."
            ),
        )
        review_record = request_review_and_wait(
            image_layer=seg_input_layer,
            labels_layer=seg_result["labels_layer"],
            timeout=review_timeout_s,
        )
        action = review_record.get("action")
        if action == "skip":
            return {
                "ok": False,
                "stage": "review_skipped",
                "error": "user skipped this sample at the review checkpoint",
                "target_channel": target_layer,
                "target_source": resolution.source,
                "labels_layer": seg_result["labels_layer"],
                "preprocess": pre_step,
                "segmentation_method": method,
                "review": review_record,
                "warnings": warnings,
            }
        if action == "timeout":
            warnings.append(
                "interactive review timed out; measuring auto-segmented labels"
            )
        # action == "commit" → labels layer was updated in-place by the dock.

    report_progress(
        stage="measurement",
        message=f"Measuring {seg_result['labels_layer']}.",
    )
    raise_if_cancelled()
    measure_result = _measure.measure_intensity(
        labels_layer=seg_result["labels_layer"],
        image_layers=[seg_input_layer],
    )
    if not measure_result.get("has_physical_units"):
        warnings.append(
            "no voxel size on the target layer — physical-unit columns were not "
            "added. Annotate or reload with scale information for area_um2 / "
            "volume_um3."
        )

    voxel = measure_result.get("voxel_scale")
    if voxel and len(voxel) == 3 and voxel[0] != voxel[1]:
        warnings.append(
            f"anisotropic voxel spacing (z={voxel[0]:.3g}, y={voxel[1]:.3g}, "
            f"x={voxel[2]:.3g}); 3D segmentation/measurement may be biased."
        )

    bundle_path: Path | None = None
    bundle_outputs = _empty_bundle_outputs()
    if domain_strategy is None:
        bundle_path, own_bundle, bundle_batch_managed, bundle_outputs, bundle_warnings = (
            _write_analysis_bundle_outputs(
                bundle=analysis_bundle,
                target_layer=target_layer,
                table_names=[measure_result["table_name"]],
                labels_cells=seg_result["labels_layer"],
                qc_png=seg_result.get("qc_png_path"),
                sample_summary={
                    "n_cells": int(
                        seg_result.get("n_objects", seg_result.get("n_cells", 0))
                    ),
                    "qc_warnings": list(seg_result.get("qc_warnings", [])),
                },
            )
        )
        warnings.extend(bundle_warnings)
        qc_png_path = _bundle_qc_png_path(
            bundle_path,
            bundle_outputs,
            seg_result.get("qc_png_path"),
        )

    if domain_strategy is not None:
        if domain_strategy != "noise_floor":
            raise ValueError(
                f"domain_strategy must be 'noise_floor' (got {domain_strategy!r})"
            )

        import pandas as _pd
        domain_layer = pre_computed_domain_layer
        domain_layer_md = dict(getattr(_get_layer(domain_layer), "metadata", {}) or {})
        domain_result = {
            "labels_layer": domain_layer,
            "n_components": int(domain_layer_md.get("n_components", 0)),
            "domain_label_count": int(domain_layer_md.get("domain_label_count", 0)),
            "domain_area_um2": float(domain_layer_md.get("domain_area_um2", 0.0)),
            "domain_volume_um3": domain_layer_md.get("domain_volume_um3"),
            "domain_voxels": int(domain_layer_md.get("domain_voxels", 0)),
            "counterstain_warnings": list(domain_layer_md.get("counterstain_warnings", [])),
            "domain_warnings": list(domain_layer_md.get("domain_warnings", [])),
        }

        domain_measure = _measure.measure_intensity(
            labels_layer=domain_layer,
            image_layers=[seg_input_layer],
        )
        domain_table_name = domain_measure["table_name"]
        cells_table_name = measure_result["table_name"]


        domain_df = get_table(domain_table_name).copy()
        cells_df = get_table(cells_table_name).copy()
        domain_df["tier"] = "domain"
        cells_df["tier"] = "cells"
        combined = _pd.concat([domain_df, cells_df], ignore_index=True, sort=False)

        tier_table_name = put_table(
            f"{target_layer}_two_tier",
            combined,
            spec={
                "tool": "analyze_target_cells",
                "mode": "two_tier",
                "target_channel": target_layer,
                "domain_layer": domain_layer,
                "cells_layer": seg_result["labels_layer"],
            },
        )

        bundle_path, own_bundle, bundle_batch_managed, bundle_outputs, bundle_warnings = (
            _write_analysis_bundle_outputs(
                bundle=analysis_bundle,
                target_layer=target_layer,
                table_names=[tier_table_name],
                labels_cells=seg_result["labels_layer"],
                labels_domain=domain_layer,
                qc_png=seg_result.get("qc_png_path"),
                sample_summary={
                    "n_cells": int(
                        seg_result.get("n_objects", seg_result.get("n_cells", 0))
                    ),
                    "n_domain_components": domain_result["n_components"],
                    "domain_label_count": domain_result["domain_label_count"],
                    "domain_area_um2": domain_result["domain_area_um2"],
                    "domain_volume_um3": domain_result["domain_volume_um3"],
                    "domain_voxels": domain_result["domain_voxels"],
                    "qc_warnings": (
                        list(seg_result.get("qc_warnings", []))
                        + list(domain_result.get("counterstain_warnings", []))
                        + list(domain_result.get("domain_warnings", []))
                    ),
                },
            )
        )
        warnings.extend(bundle_warnings)
        qc_png_path = _bundle_qc_png_path(
            bundle_path,
            bundle_outputs,
            seg_result.get("qc_png_path"),
        )

        _two_tier_result = {
            "ok": True,
            "target_channel": target_layer,
            "target_source": resolution.source,
            "preprocess": pre_step,
            "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
            "segmentation_method": method,
            "analysis_dim": "3d" if use_3d else "2d",
            "labels_layer": seg_result["labels_layer"],
            "cells_layer": seg_result["labels_layer"],
            "domain_layer": domain_layer,
            "n_domain_components": domain_result["n_components"],
            "domain_label_count": domain_result["domain_label_count"],
            "domain_area_um2": domain_result["domain_area_um2"],
            "domain_volume_um3": domain_result["domain_volume_um3"],
            "domain_voxels": domain_result["domain_voxels"],
            "n_cells": int(seg_result.get("n_objects", 0)),
            "segmentation_threshold_scope": seg_result.get("threshold_scope"),
            "tier_table_name": tier_table_name,
            "primary_table_name": tier_table_name,
            "table_name": measure_result["table_name"],
            "table_columns": measure_result["columns"],
            "qc_png_path": qc_png_path,
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "result_bundle_path": (
                None if bundle_batch_managed else str(bundle_path)
            ),
            "bundle_created": own_bundle,
            "result_files": dict(bundle_outputs),
            "voxel_scale": voxel,
            "warnings": (
                warnings
                + list(domain_result.get("counterstain_warnings", []))
                + list(domain_result.get("domain_warnings", []))
            ),
        }
        if not batch_managed:
            _record_interactive_run(
                analysis_file_key, analysis_recipe_key, "complete", _two_tier_result
            )
        return _two_tier_result

    _single_result = {
        "ok": True,
        "target_channel": target_layer,
        "target_source": resolution.source,
        "preprocess": pre_step,
        "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
        "segmentation_method": method,
        "analysis_dim": "3d" if use_3d else "2d",
        "labels_layer": seg_result["labels_layer"],
        "qc_png_path": _bundle_qc_png_path(
            bundle_path,
            bundle_outputs,
            seg_result.get("qc_png_path"),
        ),
        "qc_png_error": seg_result.get("qc_png_error"),
        "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
        "object_unit": seg_result.get("object_unit", "object_or_roi"),
        "n_objects": int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
        "do_3D": bool(use_3d),
        "object_area_min": seg_result.get("object_area_min"),
        "object_area_median": seg_result.get("object_area_median"),
        "object_area_max": seg_result.get("object_area_max"),
        "top_bright_outside_fraction": seg_result.get("top_bright_outside_fraction"),
        "mask_fraction": seg_result.get("mask_fraction"),
        "segmentation_threshold_scope": seg_result.get("threshold_scope"),
        "segmentation_warnings": seg_result.get("qc_warnings", []),
        "table_name": measure_result["table_name"],
        "primary_table_name": measure_result["table_name"],
        "table_columns": measure_result["columns"],
        "result_bundle_path": (
            None if bundle_batch_managed else str(bundle_path)
        ),
        "bundle_created": own_bundle,
        "result_files": dict(bundle_outputs),
        "voxel_scale": voxel,
        "has_physical_units": bool(measure_result.get("has_physical_units")),
        "review": review_record,
        "warnings": warnings,
    }
    if not batch_managed:
        _record_interactive_run(
            analysis_file_key, analysis_recipe_key, "complete", _single_result
        )
    return _single_result


@tool(
    description="Apply a stored analysis recipe to one or more annotated samples. "
    "Iterates samples one by one: resolves the target channel/layer, runs the "
    "Phase-2 analyze_target_cells pipeline, attaches sample/group/file columns to "
    "the resulting measurement table, records a per-sample AnalysisRun, and by "
    "default removes layers created for each sample so large batches do not retain "
    "all image volumes in memory. A failure on one sample never aborts the batch.",
    phase="3",
    worker=True,
)
def run_recipe_on_samples(
    recipe_name: str,
    sample_names: list[str] | None = None,
    execution_mode: str = "serial_cleanup",
    auto_load_files: bool = True,
    keep_layers: bool = False,
    keep_failed_layers: bool = False,
) -> dict[str, Any]:
    return BatchRecipeRunner(
        recipe_name=recipe_name,
        sample_names=sample_names,
        execution_mode=execution_mode,
        auto_load_files=auto_load_files,
        keep_layers=keep_layers,
        keep_failed_layers=keep_failed_layers,
    ).run()
