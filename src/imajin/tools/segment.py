from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.state import get_layer, get_viewer
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


@tool(
    description="Segment cells with Cellpose-SAM (generalist pretrained model). "
    "Works on 2D images and 3D z-stacks. Produces a Labels layer. Use diameter=None "
    "for auto-estimation; otherwise specify approximate cell diameter in pixels.",
    phase="2",
    vision_hint=True,
)
def cellpose_sam(
    image_layer: str,
    do_3D: bool = False,
    diameter: float | None = None,
    model: str = "cpsam",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> dict[str, Any]:
    L = get_layer(image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)

    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"cellpose_sam expects a 2D or 3D layer, got shape {data.shape}. "
            "Use orthogonal_views or pick a single channel/time slice first."
        )

    cp = _get_cellpose_model(model)
    masks, _flows, _styles = cp.eval(
        data,
        diameter=diameter,
        do_3D=do_3D and data.ndim == 3,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = np.asarray(masks).astype(np.int32)

    n_cells = int(masks.max())
    viewer = get_viewer()
    out_name = f"{L.name}_masks"
    layer = viewer.add_labels(
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "model": model,
            "diameter": diameter,
            "do_3D": do_3D and data.ndim == 3,
            "n_cells": n_cells,
        },
    )

    return {
        "labels_layer": layer.name,
        "n_cells": n_cells,
        "shape": tuple(int(s) for s in masks.shape),
        "model": model,
        "diameter": diameter,
        "do_3D": do_3D and data.ndim == 3,
    }
