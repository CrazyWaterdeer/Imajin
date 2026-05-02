from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
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
    if isinstance(md, dict):
        axes = md.get("axes")
        if isinstance(axes, str):
            stripped = axes.replace("C", "")
            if len(stripped) == ndim:
                return stripped
    if ndim == 4:
        return "TZYX"
    if ndim == 3:
        return "TYX"
    if ndim == 2:
        return "YX"
    return "".join(f"A{i}" for i in range(ndim))


def _label_qc(masks: np.ndarray) -> dict[str, Any]:
    """Lightweight quality-control summary for a labels array."""
    n = int(masks.max()) if masks.size else 0
    qc: dict[str, Any] = {
        "n_objects": n,
        "shape": tuple(int(s) for s in masks.shape),
        "dtype": str(masks.dtype),
        "empty_mask": n == 0,
    }
    if n == 0:
        qc.update(
            {
                "object_area_min": 0,
                "object_area_median": 0,
                "object_area_max": 0,
            }
        )
        return qc
    flat = masks.ravel()
    counts = np.bincount(flat, minlength=n + 1)[1:]
    qc["object_area_min"] = int(counts.min())
    qc["object_area_median"] = float(np.median(counts))
    qc["object_area_max"] = int(counts.max())
    return qc


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
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)

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
    masks, _flows, _styles = cp.eval(
        data,
        diameter=diameter,
        do_3D=use_3d,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = np.asarray(masks).astype(np.int32)
    qc = _label_qc(masks)

    out_name = f"{L.name}_masks"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "segmentation_method": "cellpose_sam",
            "model": model,
            "diameter": diameter,
            "do_3D": use_3d,
            "axes": "ZYX" if use_3d else "YX",
            **qc,
        },
    )

    return {
        "labels_layer": layer.name,
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "model": model,
        "diameter": diameter,
        "do_3D": use_3d,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
    }
