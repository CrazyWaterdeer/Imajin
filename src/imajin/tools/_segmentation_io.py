"""Shared input-side scaffolding for the segmentation tools.

Sibling of ``_segmentation_outputs.py`` (the low-level QC-PNG writer). Holds the
copy-pasted wrapper boilerplate the eight ``tools/segment`` tools used to repeat:
layer load + axis/dim guard (this module), and — added in later commits of the
GitHub issue #3 split — boundary resolution, the physical ``min_size`` fallback,
the Cellpose model accessor, the QC-PNG orchestrator, and the outline projection.
"""
from __future__ import annotations

from typing import Any

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.tools.napari_ops import snapshot_layer


def layer_axes_for_seg(layer: Any, ndim: int) -> str:
    md = getattr(layer, "metadata", None) or {}
    return layer_axes_from_metadata(md, ndim, default_3d="ZYX")


def load_and_guard(
    image_layer: str,
    *,
    tool_name: str,
    dims: str,
    ts_hint: str = "",
    ndim_hint: str = "",
):
    """Snapshot ``image_layer``, materialize it, resolve axes, and reject
    time-series / wrong-dimensionality inputs with each tool's exact message.

    ``dims`` selects the dimensionality guard + message:

    * ``"2d_or_3d"`` — reject ndim outside 2/3 (``ndim_hint`` is appended to the
      standard message, e.g. cellpose's " Reduce to YX/ZYX before calling.").
    * ``"2d_or_3d_terse"`` — the terser message ``segment_expression_domain`` uses.
    * ``"3d_only"`` — require a 3D ZYX layer (``segment_3d_cells_auto``).

    Returns ``(snapshot, data, axes)``.
    """
    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    axes = layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"{tool_name} refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). {ts_hint}".rstrip()
        )
    if dims == "3d_only":
        if data.ndim != 3 or "Z" not in axes:
            raise ValueError(
                f"{tool_name} expects a 3D ZYX layer, got shape "
                f"{data.shape} with axes {axes!r}."
            )
    elif dims == "2d_or_3d_terse":
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(
                f"{tool_name} expects 2D (YX) or 3D (ZYX), got {data.shape}."
            )
    elif dims == "2d_or_3d":
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(
                f"{tool_name} expects a 2D (YX) or 3D (ZYX) layer, got "
                f"shape {data.shape}.{ndim_hint}"
            )
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown dims={dims!r}")
    return L, data, axes


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
