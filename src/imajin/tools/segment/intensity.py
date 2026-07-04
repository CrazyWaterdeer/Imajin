from __future__ import annotations

from typing import Any

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.segmentation import (
    label_qc as _label_qc,
    label_qc_warnings as _label_qc_warnings,
    min_size_from_physical as _min_size_from_physical,
    segment_connected_regions as _segment_connected_regions,
    voxel_spacing as _voxel_spacing,
)
from imajin.tools._segmentation_io import finalize_qc_png, load_and_guard
from imajin.tools._segmentation_outputs import _source_metadata_from_layer
from imajin.tools.napari_ops import add_labels_from_worker
from imajin.tools.registry import tool


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
