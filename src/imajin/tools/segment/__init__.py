from __future__ import annotations

from typing import Any

import numpy as np

from imajin.session import get_layer, get_viewer
from imajin.analysis.arrays import materialize_array
from imajin.analysis.domain_segmentation import (
    domain_min_size_from_physical as _domain_min_size_from_physical,
    domain_physical_sizes as _domain_physical_sizes,
    filter_domain_components as _filter_domain_components,
    smooth_domain_image as _smooth_domain_image,
)
from imajin.analysis.segmentation import (
    boundary_bbox_slices as _boundary_bbox_slices,
    dilate_binary_um as _dilate_binary_um,
    intersect_labels_with_mask as _intersect_labels_with_mask,
    label_qc as _label_qc,
    scatter_labels_to_full as _scatter_labels_to_full,
    label_qc_warnings as _label_qc_warnings,
    min_size_from_physical as _min_size_from_physical,
    remove_small_binary_objects as _remove_small_binary_objects,
    segment_connected_regions as _segment_connected_regions,
    threshold_noise_floor as _threshold_noise_floor,
    voxel_spacing as _voxel_spacing,
)
from imajin.analysis.target_pipeline import (
    auto_correct_target as _auto_correct_target,
    prepare_corrected as _prepare_corrected,
    threshold_and_label as _threshold_and_label,
)
from imajin.analysis.roi_quality import assess_roi as _assess_roi
from imajin.analysis.segmentation_auto3d import (
    SegmentationCandidate as _SegmentationCandidate,
    build_auto3d_candidates as _build_auto3d_candidates,
    filter_labels_by_z_extent as _filter_labels_by_z_extent,
    rank_segmentation_labels as _rank_segmentation_labels,
    selection_confidence as _selection_confidence,
)
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools import _segmentation_io as _seg_io
from imajin.tools._segmentation_io import (
    boundary_broadcast_warning,
    effective_target_min_size,
    finalize_qc_png,
    load_and_guard,
    project_boundary_outline_2d,
    resolve_boundary,
)
from imajin.tools._segmentation_outputs import (
    _saturation_warnings,
    _source_metadata_from_layer,
    _write_segmentation_qc_png,
)
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool

# Tool families live in submodules; import them for @tool registration and to
# re-export the public tool names on ``imajin.tools.segment``.
from imajin.tools.segment.cellpose import cellpose_sam  # noqa: F401

# The Cellpose model cache lives in _segmentation_io. cellpose_sam /
# segment_3d_cells_auto call it module-qualified (_seg_io._get_cellpose_model) so a
# test patching imajin.tools._segmentation_io._get_cellpose_model intercepts the
# call after the Phase 2 package split. The bare alias below is readable only.
_get_cellpose_model = _seg_io._get_cellpose_model


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _candidate_summary(candidate: _SegmentationCandidate) -> dict[str, Any]:
    return _json_ready(
        {
            "name": candidate.name,
            "strategy": candidate.strategy,
            "score": candidate.score,
            "params": candidate.params,
            "metrics": candidate.metrics,
            "warnings": candidate.warnings,
        }
    )


@tool(
    description="Automatically choose a 3D cell/ROI segmentation for a Z-stack. "
    "Runs direct 3D target-object segmentation and plane-wise 2D segmentation with "
    "z-stitching, ranks candidates with deterministic QC metrics, and returns one "
    "ZYX Labels layer for 3D voxel-only measurement. Projection is not used for "
    "quantification. Cellpose-SAM can be included as an optional candidate.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def segment_3d_cells_auto(
    image_layer: str,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    boundary_mask: str | None = None,
    candidate_modes: list[str] | None = None,
    max_candidates: int = 8,
    stitch_min_overlap: float = 0.2,
    stitch_max_centroid_distance: float | None = None,
    stitch_max_area_ratio: float = 3.0,
    min_z_planes: int | None = 2,
    include_cellpose_sam: bool = False,
    cellpose_model: str = "cpsam",
    cellpose_diameter: float | None = None,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
    cellpose_max_size_fraction: float = 0.4,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    L, data, axes = load_and_guard(
        image_layer,
        tool_name="segment_3d_cells_auto",
        dims="3d_only",
        ts_hint="Extract a timepoint or run a per-frame workflow first.",
    )
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    boundary_data_bool, _boundary_raw = resolve_boundary(boundary_mask, raw.shape)
    _bcast = boundary_broadcast_warning(boundary_data_bool, _boundary_raw)
    if _bcast:
        saturation_warnings.append(_bcast)

    base_options = {
        "background_radius": background_radius,
        "background_method": background_method,
        "background_percentile": background_percentile,
        "threshold_method": threshold_method,
        "threshold_percentile": threshold_percentile,
        "min_snr": min_snr,
        "high_snr": high_snr,
        "min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
    }
    candidates = _build_auto3d_candidates(
        raw,
        spacing=spacing,
        base_options=base_options,
        candidate_modes=candidate_modes,
        boundary_mask=boundary_data_bool,
        max_candidates=max(1, int(max_candidates)),
        stitch_min_overlap=stitch_min_overlap,
        stitch_max_centroid_distance=stitch_max_centroid_distance,
        stitch_max_area_ratio=stitch_max_area_ratio,
        min_z_planes=min_z_planes,
    )

    if include_cellpose_sam:
        cp = _seg_io._get_cellpose_model(cellpose_model)
        anisotropy = None
        scale = tuple(float(s) for s in getattr(L, "scale", ()) or ())
        if len(scale) >= 3 and scale[1] > 0:
            anisotropy = float(scale[0] / scale[1])
        masks, _flows, _styles = cp.eval(
            raw,
            diameter=cellpose_diameter,
            do_3D=True,
            z_axis=0,
            anisotropy=anisotropy,
            flow_threshold=cellpose_flow_threshold,
            cellprob_threshold=cellpose_cellprob_threshold,
            min_size=max(1, int(min_size or 15)),
            max_size_fraction=cellpose_max_size_fraction,
        )
        cellpose_labels = np.asarray(masks, dtype=np.int32)
        cellpose_labels, z_filter = _filter_labels_by_z_extent(
            cellpose_labels,
            min_z_planes=min_z_planes,
        )
        if boundary_data_bool is not None:
            # Cellpose runs on the full raw stack; clip it to the ROI so an
            # out-of-boundary Cellpose win cannot keep labels outside the region.
            cellpose_labels = _intersect_labels_with_mask(
                cellpose_labels, boundary_data_bool, renumber=True
            )
        cp_metrics, cp_warnings, cp_score = _rank_segmentation_labels(raw, cellpose_labels)
        candidates.append(
            _SegmentationCandidate(
                name="cellpose_sam_3d",
                strategy="cellpose_sam_3d",
                labels=cellpose_labels,
                params={
                    "model": cellpose_model,
                    "diameter": cellpose_diameter,
                    "flow_threshold": cellpose_flow_threshold,
                    "cellprob_threshold": cellpose_cellprob_threshold,
                    "anisotropy": anisotropy,
                    "z_extent_filter": z_filter,
                },
                metrics=cp_metrics,
                warnings=cp_warnings,
                score=cp_score,
            )
        )
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[
            : max(1, int(max_candidates))
        ]

    if not candidates:
        raise ValueError("no segmentation candidates were generated")

    best = candidates[0]
    qc = _label_qc(best.labels)
    confidence = _selection_confidence(candidates)
    qc_warnings = saturation_warnings + list(best.warnings)
    if confidence == "low":
        qc_warnings.append(
            "automatic candidate selection confidence is low; inspect the QC image "
            "or compare candidate summaries before batch use"
        )

    out_name = f"{L.name}_auto3d_cells"
    layer = call_on_main(
        add_labels_from_worker,
        best.labels,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "auto_3d_cells",
            "selected_strategy": best.strategy,
            "selection_confidence": confidence,
            # Alias to the shared vision-gate key (M1) so an ambiguous 3D
            # candidate also surfaces its QC overlay to the agent.
            "roi_confidence": "low" if confidence == "fail" else confidence,
            "selected_score": best.score,
            "candidate_summaries": [_candidate_summary(c) for c in candidates],
            "candidate_modes": candidate_modes or ["direct_3d", "plane_stitch"],
            "include_cellpose_sam": include_cellpose_sam,
            "boundary_mask": boundary_mask,
            "min_z_planes": min_z_planes,
            "voxel_spacing": list(spacing) if spacing is not None else None,
            "axes": "ZYX",
            "qc_warnings": qc_warnings,
            **qc,
            **best.metrics,
        },
    )

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        raw,
        best.labels,
        layer,
        L,
        method="auto_3d_cells",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
    )

    return {
        "labels_layer": layer.name,
        "segmentation_method": "auto_3d_cells",
        "selected_strategy": best.strategy,
        "selection_confidence": confidence,
        "roi_confidence": "low" if confidence == "fail" else confidence,
        "selected_score": float(best.score),
        "candidate_summaries": [_candidate_summary(c) for c in candidates],
        "n_objects": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        "mask_fraction": best.metrics.get("mask_fraction"),
        "top_bright_outside_fraction": best.metrics.get("top_bright_outside_fraction"),
        "single_plane_object_fraction": best.metrics.get("single_plane_object_fraction"),
        "z_gap_object_fraction": best.metrics.get("z_gap_object_fraction"),
        "boundary_mask": boundary_mask,
        "min_z_planes": min_z_planes,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Segment bright reporter-positive cells or regions by intensity "
    "thresholding. This is often more appropriate than Cellpose-SAM for CaLexA/GCaMP "
    "or other reporter images where true cell boundaries are not visible. The output "
    "is Labels matching the input layer shape. Use split_touching=True to watershed "
    "touching blobs into candidate cells; leave it False when merged clusters should "
    "be measured as region-level ROIs.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def segment_intensity_regions(
    image_layer: str,
    threshold_method: str = "otsu",
    percentile: float = 99.0,
    min_size: int = 128,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    L, data, axes = load_and_guard(
        image_layer,
        tool_name="segment_intensity_regions",
        dims="2d_or_3d",
        ts_hint="Use extract_timepoint or a per-frame workflow first.",
    )
    spacing = _voxel_spacing(tuple(L.scale), data.ndim)
    effective_min_size = _min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=data.ndim,
    ) or int(min_size)

    masks, threshold = _segment_connected_regions(
        data,
        threshold_method=threshold_method,
        percentile=percentile,
        min_size=effective_min_size,
        smoothing_sigma=smoothing_sigma,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )
    qc = _label_qc(masks)
    qc_warnings = _label_qc_warnings(masks)

    out_name = f"{L.name}_regions"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "intensity_regions",
            "threshold_method": threshold_method,
            "threshold": threshold,
            "percentile": percentile,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "min_area_um2": min_area_um2,
            "min_volume_um3": min_volume_um3,
            "smoothing_sigma": smoothing_sigma,
            "fill_holes": fill_holes,
            "split_touching": split_touching,
            "min_distance": min_distance,
            "min_distance_um": min_distance_um,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            **qc,
        },
    )

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        data,
        masks,
        layer,
        L,
        method="intensity_regions",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
    )

    return {
        "labels_layer": layer.name,
        "n_regions": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "threshold_method": threshold_method,
        "threshold": threshold,
        "percentile": percentile,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Default target-channel segmentation for routine intensity analysis. "
    "Segments target-positive objects/ROIs from a 2D image or 3D z-stack using local "
    "background correction, background-corrected signal thresholding, size cleanup, "
    "and optional watershed splitting. This tool does not try to decide whether the "
    "objects are cells, nuclei, membranes, or clusters; it returns neutral measured "
    "objects suitable for intensity/area/count analysis.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def segment_target_objects(
    image_layer: str,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    threshold_clip_percentile: float | None = None,
    auto_mask_hyperbright: bool = False,
    hyperbright_percentile: float = 99.5,
    hyperbright_dilate_radius: int = 2,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
    boundary_mask: str | None = None,
) -> dict[str, Any]:
    L, data, axes = load_and_guard(
        image_layer,
        tool_name="segment_target_objects",
        dims="2d_or_3d",
        ts_hint="Use extract_timepoint or a per-frame workflow first.",
    )
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    effective_min_size = effective_target_min_size(
        raw,
        min_size=min_size,
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
    )
    # Load + resolve the boundary BEFORE background correction, so the expensive
    # pipeline can run on just the ROI bounding box when that is safe.
    boundary_data_bool, _boundary_raw = resolve_boundary(boundary_mask, raw.shape)
    _bcast = boundary_broadcast_warning(boundary_data_bool, _boundary_raw)
    if _bcast:
        saturation_warnings.append(_bcast)

    # Crop to the ROI bbox only when the background is a *local* operator
    # (radius > 0) and no *global* hyperbright mask is requested -- then the label
    # mask inside the ROI is identical to the full-frame result, just faster. The
    # margin (2*radius for opening + 4*sigma gaussian kernel + pad) keeps the
    # corrected image exact inside the ROI. min_size stays full-frame-derived.
    crop_slices = None
    if (
        boundary_data_bool is not None
        and _boundary_raw is not None
        and background_radius > 0
        and not auto_mask_hyperbright
    ):
        yx2d = (
            (_boundary_raw > 0)
            if _boundary_raw.ndim == 2
            else np.any(_boundary_raw > 0, axis=0)
        )
        margin = 2 * int(background_radius) + int(np.ceil(4.0 * float(smoothing_sigma))) + 8
        crop_slices = _boundary_bbox_slices(yx2d, raw.shape, margin)
        if crop_slices is not None:
            saturation_warnings.append(
                "segmentation computed inside the ROI bounding box (cropped) for speed"
            )

    raw_work = raw[crop_slices] if crop_slices is not None else raw
    boundary_work = (
        boundary_data_bool[crop_slices]
        if (crop_slices is not None and boundary_data_bool is not None)
        else boundary_data_bool
    )

    corrected_for_threshold = _prepare_corrected(
        raw_work,
        background_radius=background_radius,
        background_method=background_method,
        background_percentile=background_percentile,
        smoothing_sigma=smoothing_sigma,
    )

    # Pure threshold -> label -> QC -> score (shared with the auto-correct loop and
    # headless tests). The caller owns saturation warnings and the boundary lookup.
    seg = _threshold_and_label(
        corrected_for_threshold,
        raw_work,
        spacing=spacing,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        threshold_clip_percentile=threshold_clip_percentile,
        auto_mask_hyperbright=auto_mask_hyperbright,
        hyperbright_percentile=hyperbright_percentile,
        hyperbright_dilate_radius=hyperbright_dilate_radius,
        min_snr=min_snr,
        high_snr=high_snr,
        min_size=effective_min_size,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        boundary_mask=boundary_work,
    )
    if crop_slices is not None:
        # Place the cropped labels back into the full frame and recompute label QC
        # there so shape / n_objects / areas are correct for the layer; signal_qc
        # (mask_fraction, inside/outside separation) stays ROI-local by design.
        masks = _scatter_labels_to_full(seg.masks, raw.shape, crop_slices)
        qc = _label_qc(masks)
    else:
        masks = seg.masks
        qc = seg.qc
    threshold = seg.threshold
    high_threshold = seg.high_threshold
    noise_sigma = seg.noise_sigma
    threshold_scope = seg.threshold_scope
    signal_qc = seg.signal_qc
    qc_warnings = saturation_warnings + seg.threshold_warnings + seg.qc_warnings
    roi_score = seg.roi_score
    # v2.1: context-aware, evidence-based confidence (overrides the v1 structural
    # tier). Single-shot target objects are always the "blob" class.
    _v2 = _assess_roi(masks, spacing, roi_score, seg.score_metrics, obj_class="blob")
    roi_confidence = _v2["roi_confidence"]
    distribution_flag = _v2["distribution_flag"]
    confidence_drivers = _v2["confidence_drivers"]

    out_name = f"{L.name}_objects"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "target_objects",
            "object_unit": "object_or_roi",
            "background_radius": background_radius,
            "background_method": background_method,
            "background_percentile": background_percentile,
            "threshold_method": threshold_method,
            "threshold": threshold,
            "high_threshold": high_threshold,
            "threshold_percentile": threshold_percentile,
            "min_snr": min_snr,
            "high_snr": high_snr,
            "noise_sigma": noise_sigma,
            "threshold_scope": threshold_scope,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "min_area_um2": min_area_um2,
            "min_volume_um3": min_volume_um3,
            "smoothing_sigma": smoothing_sigma,
            "fill_holes": fill_holes,
            "split_touching": split_touching,
            "min_distance": min_distance,
            "min_distance_um": min_distance_um,
            "boundary_mask": boundary_mask,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            "roi_score": roi_score,
            "roi_confidence": roi_confidence,
            "distribution_flag": distribution_flag,
            "confidence_drivers": confidence_drivers,
            **qc,
            **signal_qc,
        },
    )

    secondary_mask_array: np.ndarray | None = None
    if boundary_mask is not None:
        bm_snapshot = call_on_main(snapshot_layer, boundary_mask)
        bm_bool = materialize_array(bm_snapshot.data) > 0
        # QC overlay is drawn on the 2D projection; keep the outline 2D.
        secondary_mask_array = project_boundary_outline_2d(bm_bool)

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        raw,
        masks,
        layer,
        L,
        method="target_objects",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
        secondary_outline_mask=secondary_mask_array,
    )

    return {
        "labels_layer": layer.name,
        "object_unit": "object_or_roi",
        "n_objects": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "background_radius": background_radius,
        "background_method": background_method,
        "background_percentile": background_percentile,
        "threshold_method": threshold_method,
        "threshold": threshold,
        "high_threshold": high_threshold,
        "threshold_percentile": threshold_percentile,
        "min_snr": min_snr,
        "high_snr": high_snr,
        "noise_sigma": noise_sigma,
        "threshold_scope": threshold_scope,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
        "boundary_mask": boundary_mask,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        **signal_qc,
        "roi_score": roi_score,
        "roi_confidence": roi_confidence,
        "distribution_flag": distribution_flag,
        "confidence_drivers": confidence_drivers,
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Hands-off target-object segmentation that auto-corrects the ROI. "
    "Runs the same pipeline as segment_target_objects, then deterministically "
    "re-thresholds to fix a too-wide or too-narrow ROI (raising/lowering the SNR "
    "bar, masking hyper-bright debris, widening the background estimate) until the "
    "ROI-quality score is confident or the iteration budget is spent. Opt-in "
    "alternative to segment_target_objects for batch/non-interactive accuracy; "
    "returns the best ROI, the applied parameters, and the correction history. "
    "Target objects only -- not for permissive expression domains, whose high "
    "coverage is expected.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def auto_segment_target(
    image_layer: str,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    max_iters: int = 3,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
    boundary_mask: str | None = None,
) -> dict[str, Any]:
    L, data, axes = load_and_guard(
        image_layer,
        tool_name="auto_segment_target",
        dims="2d_or_3d",
        ts_hint="Use extract_timepoint or a per-frame workflow first.",
    )
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    effective_min_size = effective_target_min_size(
        raw,
        min_size=min_size,
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
    )

    boundary_data_bool, _boundary_raw = resolve_boundary(boundary_mask, raw.shape)
    _bcast = boundary_broadcast_warning(boundary_data_bool, _boundary_raw)
    if _bcast:
        saturation_warnings.append(_bcast)

    params: dict[str, Any] = {
        "background_radius": background_radius,
        "background_method": background_method,
        "background_percentile": background_percentile,
        "smoothing_sigma": smoothing_sigma,
        "threshold_method": threshold_method,
        "threshold_percentile": threshold_percentile,
        "min_snr": min_snr,
        "high_snr": high_snr,
        "min_size": effective_min_size,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
    }
    best, best_params, history = _auto_correct_target(
        raw,
        spacing=spacing,
        params=params,
        boundary_mask=boundary_data_bool,
        max_iters=max_iters,
    )
    masks = best.masks
    qc = best.qc
    signal_qc = best.signal_qc
    qc_warnings = saturation_warnings + best.threshold_warnings + best.qc_warnings
    applied_params = {
        "min_snr": best_params.get("min_snr"),
        "high_snr": best_params.get("high_snr"),
        "auto_mask_hyperbright": best_params.get("auto_mask_hyperbright", False),
        "threshold_clip_percentile": best_params.get("threshold_clip_percentile"),
        "background_radius": best_params.get("background_radius"),
        "smoothing_sigma": best_params.get("smoothing_sigma"),
    }

    # v2.1 confidence, comparing the corrected mask against the raw first pass so
    # a material correction can't be silently blessed as coherent.
    raw_qc = (
        {
            "n_objects": history[0].get("n_objects", 0),
            "object_area_median": history[0].get("object_area_median"),
        }
        if history
        else None
    )
    _v2 = _assess_roi(
        masks, spacing, best.roi_score, best.score_metrics,
        obj_class="blob", raw_qc=raw_qc, corrected_qc=qc,
    )
    roi_confidence = _v2["roi_confidence"]
    distribution_flag = _v2["distribution_flag"]
    confidence_drivers = _v2["confidence_drivers"]
    correction_gap = _v2["correction_gap"]

    out_name = f"{L.name}_objects"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "auto_target_objects",
            "object_unit": "object_or_roi",
            "threshold": best.threshold,
            "high_threshold": best.high_threshold,
            "noise_sigma": best.noise_sigma,
            "threshold_scope": best.threshold_scope,
            **applied_params,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "boundary_mask": boundary_mask,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            "roi_score": best.roi_score,
            "roi_confidence": roi_confidence,
            "distribution_flag": distribution_flag,
            "confidence_drivers": confidence_drivers,
            "correction_gap": correction_gap,
            "n_iterations": len(history) - 1,
            **qc,
            **signal_qc,
        },
    )

    secondary_mask_array: np.ndarray | None = None
    if boundary_data_bool is not None:
        # boundary_data_bool may be a Z-broadcast view; project to a 2D outline so
        # the QC overlay stays small instead of materialising a Z*Y*X int32 array.
        secondary_mask_array = project_boundary_outline_2d(boundary_data_bool)

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        raw,
        masks,
        layer,
        L,
        method="auto_target_objects",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
        secondary_outline_mask=secondary_mask_array,
    )

    return {
        "labels_layer": layer.name,
        "object_unit": "object_or_roi",
        "n_objects": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "threshold": best.threshold,
        "noise_sigma": best.noise_sigma,
        "threshold_scope": best.threshold_scope,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "boundary_mask": boundary_mask,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "roi_score": best.roi_score,
        "roi_confidence": roi_confidence,
        "distribution_flag": distribution_flag,
        "confidence_drivers": confidence_drivers,
        "correction_gap": correction_gap,
        "n_iterations": len(history) - 1,
        "max_iters": max_iters,
        "applied_params": applied_params,
        "correction_history": history,
        **signal_qc,
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Segment a permissive expression domain on a reporter channel using "
    "a noise-floor threshold (median + k*MAD of the dark percentile). No background "
    "subtraction, so cluster interiors are preserved. Use as Tier 1 of two-tier "
    "expression analyses where baseline expression must be captured alongside "
    "active sub-objects. By default, connected components are cleaned and merged "
    "into one binary domain label for compact domain-level intensity measurement.",
    phase="2",
    worker=True,
)
def segment_expression_domain(
    image_layer: str,
    threshold_strategy: str = "noise_floor",
    k_mad: float = 5.25,
    dark_percentile: float = 10.0,
    counterstain_layer: str | None = None,
    counterstain_dilation_um: float = 0.0,
    is_nuclear: bool | None = None,
    min_area_um2: float = 5.0,
    min_volume_um3: float | None = None,
    smooth_sigma_um: float = 0.5,
    max_components: int = 256,
    min_component_fraction: float = 0.0,
    merge_components: bool = True,
    dilation_um: float = 0.0,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
    boundary_mask: str | None = None,
) -> dict[str, Any]:
    if threshold_strategy != "noise_floor":
        raise ValueError(
            f"threshold_strategy must be 'noise_floor' (got {threshold_strategy!r})"
        )

    L, data, axes = load_and_guard(
        image_layer,
        tool_name="segment_expression_domain",
        dims="2d_or_3d_terse",
    )
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)
    raw = np.asarray(data, dtype=np.float32)

    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    boundary_bool, _bnd_raw = resolve_boundary(boundary_mask, raw.shape)
    boundary_outline_2d: np.ndarray | None = None
    if _bnd_raw is not None:
        boundary_outline_2d = project_boundary_outline_2d(_bnd_raw > 0)

    threshold_image = _smooth_domain_image(
        raw,
        spacing=spacing,
        smooth_sigma_um=smooth_sigma_um,
    )
    if boundary_bool is not None:
        # ROI-local noise floor: estimate it from the smoothed values *inside* the ROI
        # only (finite raw + finite smoothed), so signal outside the drawn region can't
        # shift the threshold, and clip the domain to the ROI from the start.
        inside = boundary_bool & np.isfinite(raw) & np.isfinite(threshold_image)
        threshold = _threshold_noise_floor(
            threshold_image[inside], k_mad=k_mad, dark_percentile=dark_percentile
        )
        binary = inside & (threshold_image > threshold)
    else:
        threshold = _threshold_noise_floor(
            threshold_image, k_mad=k_mad, dark_percentile=dark_percentile
        )
        binary = np.isfinite(raw) & np.isfinite(threshold_image) & (
            threshold_image > threshold
        )

    counterstain_used = False
    counterstain_warnings: list[str] = []
    if counterstain_layer is not None:
        if not is_nuclear:
            counterstain_warnings.append(
                "counterstain marker is non-nuclear or unknown; reporter-only "
                "domain used"
            )
        else:
            cs_layer = call_on_main(snapshot_layer, counterstain_layer)
            cs_data = materialize_array(cs_layer.data, dtype=np.float32)
            if cs_data.shape != raw.shape:
                counterstain_warnings.append(
                    f"counterstain shape {cs_data.shape} differs from reporter "
                    f"shape {raw.shape}; counterstain ignored"
                )
            else:
                from skimage import filters as _filters
                cs_finite = cs_data[np.isfinite(cs_data)]
                if cs_finite.size and float(cs_finite.max()) > float(cs_finite.min()):
                    cs_threshold = float(_filters.threshold_otsu(cs_finite))
                    cs_binary = np.isfinite(cs_data) & (cs_data > cs_threshold)
                    if counterstain_dilation_um > 0 and spacing is not None:
                        cs_binary = _dilate_binary_um(
                            cs_binary,
                            spacing=spacing,
                            radius_um=counterstain_dilation_um,
                        )
                    binary = binary & cs_binary
                    counterstain_used = True
                else:
                    counterstain_warnings.append(
                        "counterstain has no usable signal; reporter-only domain used"
                    )

    physical_min_size = _domain_min_size_from_physical(
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
        ndim=raw.ndim,
    )
    if physical_min_size:
        binary = _remove_small_binary_objects(binary, physical_min_size)

    if dilation_um > 0 and spacing is not None:
        binary = _dilate_binary_um(binary, spacing=spacing, radius_um=dilation_um)
        if boundary_bool is not None:
            # Dilation must not grow the domain back outside the ROI.
            binary = binary & boundary_bool

    labels, component_stats, component_warnings = _filter_domain_components(
        binary,
        max_components=max_components,
        min_component_fraction=min_component_fraction,
        merge_components=merge_components,
    )
    domain_warnings = saturation_warnings + component_warnings
    size_stats = _domain_physical_sizes(labels > 0, spacing)
    n_components = int(component_stats["n_components_retained"])
    domain_label_count = int(component_stats["domain_label_count"])

    if domain_label_count == 0:
        empty = np.zeros(raw.shape, dtype=np.int32)
        out_name = f"{L.name}_domain"
        layer = call_on_main(
            add_labels_from_worker,
            empty,
            name=out_name,
            scale=tuple(L.scale),
            metadata={
                "source_layer": L.name,
                **_source_metadata_from_layer(L),
                "segmentation_method": "expression_domain",
                "noise_floor_threshold": float(threshold),
                "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
                "smooth_sigma_um": float(smooth_sigma_um),
                "k_mad": float(k_mad),
                "dark_percentile": float(dark_percentile),
                "counterstain_used": counterstain_used,
                "counterstain_warnings": counterstain_warnings,
                "domain_warnings": domain_warnings,
                "min_area_um2": float(min_area_um2),
                "min_volume_um3": min_volume_um3,
                "min_size_voxels": physical_min_size,
                "max_components": max_components,
                "min_component_fraction": float(min_component_fraction),
                "merge_components": bool(merge_components),
                **component_stats,
                **size_stats,
                "empty_mask": True,
            },
        )
        return {
            "labels_layer": layer.name,
            "n_components": 0,
            "domain_label_count": 0,
            "domain_area_um2": 0.0,
            "domain_volume_um3": 0.0 if raw.ndim == 3 else None,
            "domain_voxels": 0,
            "noise_floor_threshold": float(threshold),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "qc_png_path": None,
            "qc_png_error": None,
            "qc_png_skipped_reason": "empty mask",
            "empty_mask": True,
        }

    domain_area_um2 = float(size_stats["domain_area_um2"])
    domain_volume_um3 = size_stats["domain_volume_um3"]

    out_name = f"{L.name}_domain"
    layer = call_on_main(
        add_labels_from_worker,
        labels,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "expression_domain",
            "noise_floor_threshold": float(threshold),
            "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
            "smooth_sigma_um": float(smooth_sigma_um),
            "k_mad": float(k_mad),
            "dark_percentile": float(dark_percentile),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "n_components": n_components,
            "domain_label_count": domain_label_count,
            "domain_area_um2": domain_area_um2,
            "domain_volume_um3": domain_volume_um3,
            "domain_voxels": int(size_stats["domain_voxels"]),
            "min_area_um2": float(min_area_um2),
            "min_volume_um3": min_volume_um3,
            "min_size_voxels": physical_min_size,
            "max_components": max_components,
            "min_component_fraction": float(min_component_fraction),
            "merge_components": bool(merge_components),
            **component_stats,
            "boundary_mask": boundary_mask,
            "threshold_scope": "boundary_mask" if boundary_bool is not None else "global",
            "empty_mask": False,
        },
    )

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        raw,
        labels,
        layer,
        L,
        method="expression_domain",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
        secondary_outline_mask=boundary_outline_2d,
    )

    return {
        "labels_layer": layer.name,
        "n_components": n_components,
        "domain_label_count": domain_label_count,
        "domain_area_um2": domain_area_um2,
        "domain_volume_um3": domain_volume_um3,
        "domain_voxels": int(size_stats["domain_voxels"]),
        "noise_floor_threshold": float(threshold),
        "counterstain_used": counterstain_used,
        "counterstain_warnings": counterstain_warnings,
        "domain_warnings": domain_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
        "boundary_mask": boundary_mask,
        "threshold_scope": "boundary_mask" if boundary_bool is not None else "global",
        "empty_mask": False,
    }


@tool(
    description="Re-run target object segmentation on the same image with "
    "overridden parameters and replace the existing labels layer in place. "
    "Phase 3 of the SNR/ROI work; intended for natural-language corrections "
    "such as 'lower min_snr and turn on the hyperbright mask'. Returns the "
    "new label count, threshold, and any QC warnings.",
    phase="2",
    llm=True,
    worker=True,
)
def correct_roi(
    image_layer: str,
    labels_layer: str,
    min_snr: float | None = None,
    high_snr: float | None = None,
    threshold_method: str | None = None,
    threshold_clip_percentile: float | None = None,
    auto_mask_hyperbright: bool | None = None,
    hyperbright_percentile: float | None = None,
    background_radius: int | None = None,
    smoothing_sigma: float | None = None,
    min_size: int | None = None,
) -> dict[str, Any]:

    try:
        labels_layer_obj = call_on_main(get_layer, labels_layer)
    except KeyError:
        return {"ok": False, "error": f"labels_layer '{labels_layer}' not found"}

    prev_meta = dict(getattr(labels_layer_obj, "metadata", {}) or {})

    def _resolve(name: str, value: Any, default: Any) -> Any:
        if value is not None:
            return value
        if name in prev_meta:
            return prev_meta[name]
        return default

    kwargs: dict[str, Any] = {
        "image_layer": image_layer,
        "threshold_method": _resolve("threshold_method", threshold_method, "auto"),
        "threshold_clip_percentile": _resolve(
            "threshold_clip_percentile", threshold_clip_percentile, None
        ),
        "auto_mask_hyperbright": bool(
            _resolve("auto_mask_hyperbright", auto_mask_hyperbright, False)
        ),
        "hyperbright_percentile": float(
            _resolve("hyperbright_percentile", hyperbright_percentile, 99.5)
        ),
        "min_snr": float(_resolve("min_snr", min_snr, 2.0)),
        "high_snr": float(_resolve("high_snr", high_snr, 4.0)),
        "background_radius": int(_resolve("background_radius", background_radius, 48)),
        "smoothing_sigma": float(_resolve("smoothing_sigma", smoothing_sigma, 1.0)),
    }
    if min_size is not None:
        kwargs["min_size"] = int(min_size)
    if "boundary_mask" in prev_meta and prev_meta["boundary_mask"]:
        kwargs["boundary_mask"] = prev_meta["boundary_mask"]

    new_result = segment_target_objects(**kwargs)
    new_labels_layer = new_result.get("labels_layer")
    if not new_labels_layer:
        return {
            "ok": False,
            "error": "segment_target_objects did not produce a labels layer",
            "result": new_result,
        }

    new_layer_obj = call_on_main(get_layer, new_labels_layer)
    new_data = materialize_array(new_layer_obj.data)
    new_metadata = dict(getattr(new_layer_obj, "metadata", {}) or {})

    def _commit_to_target_layer():
        target = get_layer(labels_layer)
        target.data = np.asarray(new_data, dtype=np.int32)
        merged = dict(getattr(target, "metadata", {}) or {})
        merged.update(new_metadata)
        merged["corrected_from"] = new_labels_layer
        target.metadata = merged
        # Drop the throwaway layer the worker created.
        viewer_layers = target.parent if hasattr(target, "parent") else None
        if viewer_layers is None:
            viewer = get_viewer()
            if viewer is not None and new_labels_layer in viewer.layers:
                try:
                    viewer.layers.remove(new_labels_layer)
                except Exception:
                    pass

    call_on_main(_commit_to_target_layer)

    return {
        "ok": True,
        "labels_layer": labels_layer,
        "replaced_with": new_labels_layer,
        "n_objects": new_result.get("n_objects", new_result.get("n_cells", 0)),
        "threshold": new_result.get("threshold"),
        "threshold_scope": new_result.get("threshold_scope"),
        "qc_warnings": new_result.get("qc_warnings", []),
        # Surface the corrected result's score + overlay (H3) so the agent-vision
        # gate fires on the correction it just made -- the moment it most needs
        # to see whether the ROI is now right.
        "roi_score": new_result.get("roi_score"),
        "roi_confidence": new_result.get("roi_confidence"),
        "qc_png_path": new_result.get("qc_png_path"),
        "applied_params": {
            k: v for k, v in kwargs.items() if k not in {"image_layer"}
        },
    }


@tool(
    description="Open the interactive ROI review dock against an existing "
    "(image, labels) pair so the user can mark points/regions to add or "
    "remove on a MIP overlay and rebuild the ROI on the original 3D stack. "
    "Single-sample manual mode entry point for Phase 2 of the SNR/ROI work.",
    phase="2",
    llm=True,
    worker=False,
)
def review_target_roi(
    image_layer: str,
    labels_layer: str,
) -> dict[str, Any]:
    from imajin.ui.main import _show_review_panel

    viewer = get_viewer()
    if viewer is None:
        return {
            "ok": False,
            "error": "No napari viewer is available; review can only run "
            "inside the imajin GUI.",
        }

    if image_layer not in viewer.layers:
        return {"ok": False, "error": f"image_layer '{image_layer}' not found"}
    if labels_layer not in viewer.layers:
        return {"ok": False, "error": f"labels_layer '{labels_layer}' not found"}

    dock_widget = call_on_main(_show_review_panel, viewer)
    if dock_widget is None:
        return {"ok": False, "error": "review dock could not be opened"}

    call_on_main(dock_widget.request_layers, image_layer, labels_layer)

    return {
        "ok": True,
        "image_layer": image_layer,
        "labels_layer": labels_layer,
        "message": (
            "Review dock opened. Mark add/remove points and regions, then "
            "click Rebuild ROI; click Commit to write changes back to the "
            "labels layer."
        ),
    }
