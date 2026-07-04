from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.segmentation import (
    label_qc as _label_qc,
    label_qc_warnings as _label_qc_warnings,
)
from imajin.tools import _segmentation_io as _seg_io
from imajin.tools._segmentation_io import finalize_qc_png, load_and_guard
from imajin.tools._segmentation_outputs import _source_metadata_from_layer
from imajin.tools.napari_ops import add_labels_from_worker
from imajin.tools.registry import tool


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
    L, data, axes = load_and_guard(
        image_layer,
        tool_name="cellpose_sam",
        dims="2d_or_3d",
        ts_hint="Use extract_timepoint to pick a frame first, or run a per-frame workflow.",
        ndim_hint=" Reduce to YX/ZYX before calling.",
    )

    is_3d_input = data.ndim == 3 and "Z" in axes
    use_3d = bool(do_3D) and is_3d_input
    if do_3D and not is_3d_input:
        # Caller asked for 3D but data is 2D — fall back silently to 2D rather
        # than confusing Cellpose.
        use_3d = False

    cp = _seg_io._get_cellpose_model(model)
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

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        data,
        masks,
        layer,
        L,
        method="cellpose_sam",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
    )

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
