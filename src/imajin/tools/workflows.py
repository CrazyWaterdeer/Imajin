from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.analysis.workflow import (
    check_analysis_memory_budget as _check_analysis_memory_budget,
    decide_3d as _decide_3d,
    derive_size_params as _derive_size_params,
    normalize_domain_spec as _normalize_domain_spec,
    normalize_segmentation_method as _normalize_segmentation_method,
)
from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import (
    AmbiguousChannelError,
    get_layer as _get_layer,
    get_table,
    put_table,
    resolve_target_channel,
)
from imajin.tools.batch_runner import BatchRecipeRunner
from imajin.tools import measure as _measure
from imajin.tools._workflow_outputs import (
    _bundle_qc_png_path,
    _empty_bundle_outputs,
    _write_analysis_bundle_outputs,
)
from imajin.tools._workflow_steps import (
    _layer_axes,
    _precompute_domain_layer,
    _run_preprocess_step,
    _run_segmentation_step,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


@tool(
    description="High-level target object analysis workflow. Resolves the target "
    "channel, optionally preprocesses it, segments target-positive objects/ROIs, "
    "and measures per-object intensity and size on the same target channel. Pass "
    "target as a layer name, color (green/red/UV/IR), or marker (GFP/DAPI). Pass preprocess="
    "'rolling_ball' / 'auto_contrast' / 'gaussian_denoise' to apply one preprocessing "
    "step before segmentation. Returns labels layer, measurement table, object count, "
    "and QC metrics. Counterstain channels are not auto-selected — annotate them as "
    "'counterstain' first if you only have one target channel.",
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
    domain_strategy: str | None = None,
    domain_options: dict[str, Any] | None = None,
    counterstain_layer: str | None = None,
    cell_diameter_um: float | None = None,
    review_mode: str = "auto",
    review_timeout_s: float | None = None,
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

    seg_input_layer, pre_step, pre_record = _run_preprocess_step(
        target_layer,
        preprocess,
    )

    snapshot = call_on_main(snapshot_layer, seg_input_layer)
    axes = _layer_axes(snapshot)
    use_3d = _decide_3d(do_3D, axes, getattr(snapshot.data, "ndim", 2))

    # Tier-1: pre-compute expression domain before Tier-2 segmentation.
    pre_computed_domain_layer: str | None = None
    if domain_strategy is not None:
        pre_computed_domain_layer = _precompute_domain_layer(
            target_layer=target_layer,
            snapshot=snapshot,
            domain_options=domain_options,
            counterstain_layer=counterstain_layer,
            cell_diameter_um=cell_diameter_um,
        )

    method = _normalize_segmentation_method(segmentation_method)

    _check_analysis_memory_budget(seg_input_layer, data=snapshot.data, method=method)

    seg_options = dict(segmentation_options or {})
    seg_options.pop("tool", None)
    seg_options.pop("do_3D", None)
    seg_options.pop("diameter", None)
    if pre_computed_domain_layer is not None:
        seg_options.setdefault("boundary_mask", pre_computed_domain_layer)
        seg_options.setdefault("min_snr", 1.6)
        seg_options.setdefault("high_snr", 3.2)
    seg_result = _run_segmentation_step(
        method=method,
        seg_input_layer=seg_input_layer,
        seg_options=seg_options,
        use_3d=use_3d,
        diameter=diameter,
    )
    if seg_result.get("empty_mask", False):
        return {
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
        bundle_path, own_bundle, bundle_outputs, bundle_warnings = (
            _write_analysis_bundle_outputs(
                target_layer=target_layer,
                target_source=resolution.source,
                segmentation_method=method,
                analysis_dim="3d" if use_3d else "2d",
                tier="single_tier",
                bundle_suffix="single",
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

        bundle_path, own_bundle, bundle_outputs, bundle_warnings = (
            _write_analysis_bundle_outputs(
                target_layer=target_layer,
                target_source=resolution.source,
                segmentation_method=method,
                analysis_dim="3d" if use_3d else "2d",
                tier="two_tier",
                bundle_suffix="two_tier",
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

        return {
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
            "result_bundle_path": str(bundle_path) if own_bundle else None,
            "result_files": dict(bundle_outputs),
            "voxel_scale": voxel,
            "warnings": (
                warnings
                + list(domain_result.get("counterstain_warnings", []))
                + list(domain_result.get("domain_warnings", []))
            ),
        }

    return {
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
        "result_bundle_path": str(bundle_path) if own_bundle else None,
        "result_files": dict(bundle_outputs),
        "voxel_scale": voxel,
        "has_physical_units": bool(measure_result.get("has_physical_units")),
        "review": review_record,
        "warnings": warnings,
    }

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
