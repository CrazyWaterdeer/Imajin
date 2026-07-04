from __future__ import annotations

from typing import Any

import numpy as np

from imajin.session import get_layer, get_viewer
from imajin.analysis.arrays import materialize_array
from imajin.analysis.segmentation import (
    boundary_bbox_slices as _boundary_bbox_slices,
    label_qc as _label_qc,
    scatter_labels_to_full as _scatter_labels_to_full,
    voxel_spacing as _voxel_spacing,
)
from imajin.analysis.target_pipeline import (
    auto_correct_target as _auto_correct_target,
    prepare_corrected as _prepare_corrected,
    threshold_and_label as _threshold_and_label,
)
from imajin.analysis.roi_quality import assess_roi as _assess_roi
from imajin.agent.qt_dispatch import call_on_main
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
)
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool


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
