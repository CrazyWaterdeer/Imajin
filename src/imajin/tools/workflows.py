from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from imajin.analysis.arrays import metadata_axes_without_channel
from imajin.analysis.workflow import (
    build_sample_summary as _build_sample_summary,
    check_analysis_memory_budget as _check_analysis_memory_budget,
    decide_3d as _decide_3d,
    derive_size_params as _derive_size_params,
    normalize_domain_spec as _normalize_domain_spec,
    normalize_preprocess as _normalize_preprocess,
    normalize_segmentation_method as _normalize_segmentation_method,
)
from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import (
    AmbiguousChannelError,
    resolve_target_channel,
)
from imajin.tools.batch_runner import BatchRecipeRunner
from imajin.tools import measure as _measure
from imajin.tools import preprocess as _preprocess
from imajin.tools import segment as _segment
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


def _layer_axes(snapshot: Any) -> str | None:
    return metadata_axes_without_channel(
        snapshot.metadata if isinstance(snapshot.metadata, dict) else None,
        getattr(snapshot.data, "ndim", 0),
    )


def _filtered_kwargs(func: Any, options: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(func).parameters
    return {key: value for key, value in options.items() if key in params}


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


def _remove_copied_standalone_qc(
    qc_png: str | None,
    *,
    bundle_path: Path,
    copied_rel: str | None,
) -> None:
    if not qc_png or not copied_rel:
        return
    from imajin.paths import normalize_user_path

    src = normalize_user_path(qc_png).resolve()
    dst = (bundle_path / copied_rel).resolve()
    if src == dst or not src.exists():
        return
    if src.parent.name != "segmentation_qc":
        return
    src.unlink()
    try:
        src.parent.rmdir()
    except OSError:
        pass


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
        current_bundle,
        current_sample_slug,
        finalize_bundle_metadata,
        populate_sample_outputs,
        write_combined_csv,
    )

    warnings: list[str] = []
    sample_slug = current_sample_slug() or slugify_result_name(target_layer)
    parent = current_bundle()
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
    try:
        _remove_copied_standalone_qc(
            qc_png,
            bundle_path=bundle_path,
            copied_rel=bundle_outputs.get("qc_png"),
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"standalone QC cleanup failed: {type(exc).__name__}: {exc}"
        )

    if own_bundle:
        summary = _build_sample_summary(
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

    seg_input_layer = target_layer
    pre_step = _normalize_preprocess(preprocess)
    pre_record: dict[str, Any] | None = None
    if pre_step == "rolling_ball":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.rolling_ball_background(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "auto_contrast":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.auto_contrast(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "gaussian_denoise":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.gaussian_denoise(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]

    snapshot = call_on_main(snapshot_layer, seg_input_layer)
    axes = _layer_axes(snapshot)
    use_3d = _decide_3d(do_3D, axes, getattr(snapshot.data, "ndim", 2))

    # Tier-1: pre-compute expression domain before Tier-2 segmentation.
    pre_computed_domain_layer: str | None = None
    if domain_strategy is not None:
        from imajin.tools import channels as _channels_pre
        from imajin.tools.segment import segment_expression_domain as _seg_dom
        cs_layer_pre = counterstain_layer
        cs_is_nuclear_pre: bool | None = None
        if cs_layer_pre is None:
            cs_info_pre = _channels_pre.detect_counterstain_channel()
            if cs_info_pre["confidence"] == "annotated":
                cs_layer_pre = cs_info_pre["counterstain_layer"]
                cs_is_nuclear_pre = cs_info_pre["is_nuclear"]
        d_opts = dict(domain_options or {})
        if cs_layer_pre:
            d_opts.setdefault("counterstain_layer", cs_layer_pre)
            d_opts.setdefault("is_nuclear", cs_is_nuclear_pre)
        derived_pre = _derive_size_params(
            cell_diameter_um,
            _segment._voxel_spacing(tuple(snapshot.scale), getattr(snapshot.data, "ndim", 2)),
        )
        if "min_area_um2" not in d_opts and "min_area_um2" in derived_pre:
            d_opts["min_area_um2"] = derived_pre["min_area_um2"]
        d_opts.setdefault("k_mad", 6.0)
        d_opts.setdefault("dark_percentile", 10.0)
        d_opts.setdefault("smooth_sigma_um", 0.75)
        d_opts.setdefault("max_components", 128)
        d_opts.setdefault("save_qc_png", False)
        domain_pre = _seg_dom(
            image_layer=target_layer,
            **_filtered_kwargs(_seg_dom, d_opts),
        )
        pre_computed_domain_layer = domain_pre["labels_layer"]

    method = _normalize_segmentation_method(segmentation_method)

    _check_analysis_memory_budget(seg_input_layer, data=snapshot.data, method=method)

    report_progress(stage="segmentation", message=f"Segmenting {seg_input_layer}.")
    raise_if_cancelled()
    seg_options = dict(segmentation_options or {})
    seg_options.pop("tool", None)
    seg_options.pop("do_3D", None)
    seg_options.pop("diameter", None)
    if pre_computed_domain_layer is not None:
        seg_options.setdefault("boundary_mask", pre_computed_domain_layer)
        seg_options.setdefault("min_snr", 1.5)
        seg_options.setdefault("high_snr", 3.0)
    if method == "target_objects":
        seg_result = _segment.segment_target_objects(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_target_objects, seg_options),
        )
    elif method == "intensity_regions":
        seg_result = _segment.segment_intensity_regions(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_intensity_regions, seg_options),
        )
    else:
        cellpose_options = {
            **seg_options,
            "do_3D": use_3d,
            "diameter": diameter,
        }
        seg_result = _segment.cellpose_sam(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.cellpose_sam, cellpose_options),
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
        from imajin.agent.state import get_layer as _get_layer
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

        from imajin.agent.state import get_table, put_table

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
        "segmentation_warnings": seg_result.get("qc_warnings", []),
        "table_name": measure_result["table_name"],
        "primary_table_name": measure_result["table_name"],
        "table_columns": measure_result["columns"],
        "result_bundle_path": str(bundle_path) if own_bundle else None,
        "result_files": dict(bundle_outputs),
        "voxel_scale": voxel,
        "has_physical_units": bool(measure_result.get("has_physical_units")),
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
