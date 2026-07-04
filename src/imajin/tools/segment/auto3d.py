from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.segmentation import (
    intersect_labels_with_mask as _intersect_labels_with_mask,
    label_qc as _label_qc,
    voxel_spacing as _voxel_spacing,
)
from imajin.analysis.segmentation_auto3d import (
    SegmentationCandidate as _SegmentationCandidate,
    build_auto3d_candidates as _build_auto3d_candidates,
    filter_labels_by_z_extent as _filter_labels_by_z_extent,
    rank_segmentation_labels as _rank_segmentation_labels,
    selection_confidence as _selection_confidence,
)
from imajin.tools import _segmentation_io as _seg_io
from imajin.tools._segmentation_io import (
    boundary_broadcast_warning,
    finalize_qc_png,
    load_and_guard,
    resolve_boundary,
)
from imajin.tools._segmentation_outputs import (
    _saturation_warnings,
    _source_metadata_from_layer,
)
from imajin.tools.napari_ops import add_labels_from_worker
from imajin.tools.registry import tool


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
