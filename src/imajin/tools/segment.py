from __future__ import annotations

from typing import Any

import numpy as np

from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.analysis.domain_segmentation import (
    domain_min_size_from_physical as _domain_min_size_from_physical,
    domain_physical_sizes as _domain_physical_sizes,
    filter_domain_components as _filter_domain_components,
    smooth_domain_image as _smooth_domain_image,
)
from imajin.analysis.segmentation import (
    dilate_binary_um as _dilate_binary_um,
    estimate_local_background as _estimate_local_background,
    intersect_labels_with_mask as _intersect_labels_with_mask,
    label_qc as _label_qc,
    label_qc_warnings as _label_qc_warnings,
    labels_from_binary as _labels_from_binary,
    min_size_from_physical as _min_size_from_physical,
    remove_small_binary_objects as _remove_small_binary_objects,
    segment_connected_regions as _segment_connected_regions,
    target_object_qc as _target_object_qc,
    threshold_noise_floor as _threshold_noise_floor,
    voxel_spacing as _voxel_spacing,
)
from imajin.analysis.target_segmentation import (
    target_threshold_for_scope as _target_threshold_for_scope,
)
from imajin.analysis.segmentation_auto3d import (
    SegmentationCandidate as _SegmentationCandidate,
    build_auto3d_candidates as _build_auto3d_candidates,
    filter_labels_by_z_extent as _filter_labels_by_z_extent,
    rank_segmentation_labels as _rank_segmentation_labels,
    selection_confidence as _selection_confidence,
)
from imajin.agent.qt_dispatch import call_on_main
from imajin.paths import normalize_user_path
from imajin.tools._segmentation_outputs import (
    _default_qc_png_path,
    _saturation_warnings,
    _save_qc_png,
    _source_metadata_from_layer,
    _write_segmentation_qc_png,
)
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool

_CACHED_MODELS: dict[str, Any] = {}


def _get_cellpose_model(model_name: str = "cpsam"):
    if model_name in _CACHED_MODELS:
        return _CACHED_MODELS[model_name]
    import torch
    from cellpose import models

    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
    _CACHED_MODELS[model_name] = model
    return model


def _layer_axes_for_seg(layer: Any, ndim: int) -> str:
    md = getattr(layer, "metadata", None) or {}
    return layer_axes_from_metadata(md, ndim, default_3d="ZYX")


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
    description="Segment cells with Cellpose-SAM (generalist pretrained model). "
    "Works on 2D images (YX) and 3D z-stacks (ZYX). 4D (TZYX) and time-series (TYX) "
    "inputs must be reduced to a single timepoint first via extract_timepoint or a "
    "per-frame workflow. Set do_3D=True for true 3D segmentation on Z-stacks. "
    "Use diameter=None for auto-estimation; otherwise specify approximate cell "
    "diameter in pixels.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def cellpose_sam(
    image_layer: str,
    do_3D: bool = False,
    diameter: float | None = None,
    model: str = "cpsam",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"cellpose_sam refuses to run on a time-series layer ({axes}, "
            f"shape {data.shape}). Use extract_timepoint to pick a frame first, "
            "or run a per-frame workflow."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"cellpose_sam expects a 2D (YX) or 3D (ZYX) layer, got shape "
            f"{data.shape}. Reduce to YX/ZYX before calling."
        )

    is_3d_input = data.ndim == 3 and "Z" in axes
    use_3d = bool(do_3D) and is_3d_input
    if do_3D and not is_3d_input:
        # Caller asked for 3D but data is 2D — fall back silently to 2D rather
        # than confusing Cellpose.
        use_3d = False

    cp = _get_cellpose_model(model)
    scale = tuple(float(s) for s in getattr(L, "scale", ()) or ())
    anisotropy = None
    if use_3d and len(scale) >= 3 and scale[1] > 0:
        anisotropy = float(scale[0] / scale[1])
    masks, _flows, _styles = cp.eval(
        data,
        diameter=diameter,
        do_3D=use_3d,
        z_axis=0 if use_3d else None,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
    )
    masks = np.asarray(masks).astype(np.int32)
    qc = _label_qc(masks)
    qc_warnings = _label_qc_warnings(masks)

    out_name = f"{L.name}_masks"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "cellpose_sam",
            "model": model,
            "diameter": diameter,
            "do_3D": use_3d,
            "min_size": min_size,
            "max_size_fraction": max_size_fraction,
            "anisotropy": anisotropy,
            "axes": "ZYX" if use_3d else "YX",
            "qc_warnings": qc_warnings,
            **qc,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                data,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="cellpose_sam",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "model": model,
        "diameter": diameter,
        "do_3D": use_3d,
        "min_size": min_size,
        "max_size_fraction": max_size_fraction,
        "anisotropy": anisotropy,
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
    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_3d_cells_auto refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). Extract a timepoint or run a "
            "per-frame workflow first."
        )
    if data.ndim != 3 or "Z" not in axes:
        raise ValueError(
            f"segment_3d_cells_auto expects a 3D ZYX layer, got shape "
            f"{data.shape} with axes {axes!r}."
        )

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    boundary_data_bool: np.ndarray | None = None
    if boundary_mask is not None:
        boundary_layer_snapshot = call_on_main(snapshot_layer, boundary_mask)
        boundary_raw = materialize_array(boundary_layer_snapshot.data)
        if boundary_raw.shape != raw.shape:
            raise ValueError(
                f"boundary_mask shape {boundary_raw.shape} does not match "
                f"target image shape {raw.shape}"
            )
        boundary_data_bool = boundary_raw > 0

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
        cp = _get_cellpose_model(cellpose_model)
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

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                best.labels,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="auto_3d_cells",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "segmentation_method": "auto_3d_cells",
        "selected_strategy": best.strategy,
        "selection_confidence": confidence,
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
    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_intensity_regions refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). Use extract_timepoint or a per-frame "
            "workflow first."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"segment_intensity_regions expects a 2D (YX) or 3D (ZYX) layer, got "
            f"shape {data.shape}."
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

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                data,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="intensity_regions",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

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
    from skimage import filters

    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_target_objects refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). Use extract_timepoint or a per-frame "
            "workflow first."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"segment_target_objects expects a 2D (YX) or 3D (ZYX) layer, got "
            f"shape {data.shape}."
        )

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    xy_area = int(np.prod(raw.shape[-2:])) if raw.ndim >= 2 else int(raw.size)
    physical_min_size = _min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=raw.ndim,
    )
    effective_min_size = physical_min_size or max(16, min(512, int(round(xy_area * 0.00005))))
    background = _estimate_local_background(
        raw,
        radius=background_radius,
        method=background_method,
        percentile=background_percentile,
    )
    corrected = raw - background
    corrected[~np.isfinite(corrected)] = 0.0

    if smoothing_sigma > 0:
        sigma: float | tuple[float, ...]
        if corrected.ndim == 3:
            sigma = (0.0, float(smoothing_sigma), float(smoothing_sigma))
        else:
            sigma = float(smoothing_sigma)
        corrected_for_threshold = filters.gaussian(
            corrected,
            sigma=sigma,
            preserve_range=True,
        ).astype(np.float32)
    else:
        corrected_for_threshold = corrected

    # Load the boundary mask early so it can be used as the hysteresis grow region.
    boundary_data_bool: np.ndarray | None = None
    if boundary_mask is not None:
        boundary_layer_snapshot = call_on_main(snapshot_layer, boundary_mask)
        _boundary_raw = materialize_array(boundary_layer_snapshot.data)
        if _boundary_raw.shape != raw.shape:
            raise ValueError(
                f"boundary_mask shape {_boundary_raw.shape} does not match "
                f"target image shape {raw.shape}"
            )
        boundary_data_bool = _boundary_raw > 0

    threshold, noise_sigma, threshold_scope, threshold_warnings = _target_threshold_for_scope(
        corrected_for_threshold,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        min_snr=min_snr,
        boundary_mask=boundary_data_bool,
        clip_percentile=threshold_clip_percentile,
        auto_mask_hyperbright=auto_mask_hyperbright,
        hyperbright_percentile=hyperbright_percentile,
        hyperbright_dilate_radius=hyperbright_dilate_radius,
    )

    high_threshold = max(float(threshold), float(high_snr) * float(noise_sigma))
    if boundary_data_bool is not None:
        scoped_threshold_image = np.where(
            boundary_data_bool,
            corrected_for_threshold,
            -np.inf,
        ).astype(np.float32, copy=False)
        low_candidates = (scoped_threshold_image >= float(threshold)) & boundary_data_bool
        high_seeds = (scoped_threshold_image >= high_threshold) & boundary_data_bool
        if high_threshold > threshold and np.any(high_seeds):
            binary = (
                filters.apply_hysteresis_threshold(
                    scoped_threshold_image,
                    low=float(threshold),
                    high=float(high_threshold),
                )
                & boundary_data_bool
            )
        else:
            binary = low_candidates
    elif high_threshold > threshold and np.any(corrected_for_threshold >= high_threshold):
        binary = filters.apply_hysteresis_threshold(
            corrected_for_threshold,
            low=float(threshold),
            high=float(high_threshold),
        )
    else:
        binary = corrected_for_threshold > float(threshold)

    masks = _labels_from_binary(
        binary,
        min_size=effective_min_size,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )
    qc = _label_qc(masks)
    signal_qc, qc_warnings = _target_object_qc(
        raw,
        corrected_for_threshold,
        masks,
        noise_sigma=noise_sigma,
    )
    qc_warnings = saturation_warnings + threshold_warnings + qc_warnings

    if boundary_data_bool is not None:
        masks = _intersect_labels_with_mask(
            masks, boundary_data_bool, renumber=True
        )
        qc = _label_qc(masks)
        signal_qc, qc_warnings = _target_object_qc(
            raw,
            corrected_for_threshold,
            masks,
            noise_sigma=noise_sigma,
        )
        qc_warnings = saturation_warnings + threshold_warnings + qc_warnings

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
            **qc,
            **signal_qc,
        },
    )

    secondary_mask_array: np.ndarray | None = None
    if boundary_mask is not None:
        bm_snapshot = call_on_main(snapshot_layer, boundary_mask)
        bm_data = materialize_array(bm_snapshot.data)
        secondary_mask_array = (bm_data > 0).astype(np.int32)

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="target_objects",
                force=qc_png_path is not None,
                secondary_outline_mask=secondary_mask_array,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

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
) -> dict[str, Any]:
    if threshold_strategy != "noise_floor":
        raise ValueError(
            f"threshold_strategy must be 'noise_floor' (got {threshold_strategy!r})"
        )

    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)
    raw = np.asarray(data, dtype=np.float32)

    axes = _layer_axes_for_seg(L, raw.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_expression_domain refuses to run on a time-series layer "
            f"({axes}, shape {raw.shape})."
        )
    if raw.ndim < 2 or raw.ndim > 3:
        raise ValueError(
            f"segment_expression_domain expects 2D (YX) or 3D (ZYX), got {raw.shape}."
        )

    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    threshold_image = _smooth_domain_image(
        raw,
        spacing=spacing,
        smooth_sigma_um=smooth_sigma_um,
    )
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
            "empty_mask": False,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                labels,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="expression_domain",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

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
    from imajin.agent.state import get_layer

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
            from imajin.agent.state import get_viewer
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
    from imajin.agent.state import get_viewer

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
