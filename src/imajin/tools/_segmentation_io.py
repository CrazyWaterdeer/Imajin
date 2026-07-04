"""Shared input-side scaffolding for the segmentation tools.

Sibling of ``_segmentation_outputs.py`` (the low-level QC-PNG writer). Holds the
copy-pasted wrapper boilerplate the eight ``tools/segment`` tools used to repeat:
layer load + axis/dim guard (this module), and — added in later commits of the
GitHub issue #3 split — boundary resolution, the physical ``min_size`` fallback,
the Cellpose model accessor, the QC-PNG orchestrator, and the outline projection.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.analysis.segmentation import min_size_from_physical, resolve_boundary_mask
from imajin.paths import normalize_user_path
from imajin.tools._segmentation_outputs import _default_qc_png_path, _save_qc_png
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


def resolve_boundary(boundary_mask, raw_shape):
    """Snapshot + materialize a boundary layer and resolve it to a bool mask
    broadcast to ``raw_shape``. Returns ``(bool_mask, boundary_raw)`` or
    ``(None, None)`` when no boundary was given. The caller decides whether to
    emit the broadcast warning (see ``boundary_broadcast_warning``) and how to use
    the raw array (crop bbox / QC outline)."""
    if boundary_mask is None:
        return None, None
    snap = call_on_main(snapshot_layer, boundary_mask)
    boundary_raw = materialize_array(snap.data)
    return resolve_boundary_mask(boundary_raw, raw_shape), boundary_raw


def boundary_broadcast_warning(bool_mask, boundary_raw):
    """The "2D ROI broadcast across all N Z planes" note the target / auto-target /
    auto-3d tools append when a 2D ROI was broadcast over a 3D stack. Opt-in:
    segment_expression_domain deliberately does not emit it."""
    if (
        bool_mask is not None
        and boundary_raw is not None
        and bool_mask.ndim == 3
        and boundary_raw.ndim == 2
    ):
        return (
            f"boundary_mask is a 2D ROI broadcast across all "
            f"{bool_mask.shape[0]} Z planes"
        )
    return None


def effective_target_min_size(raw, *, min_size, min_area_um2, min_volume_um3, spacing):
    """The physical-or-heuristic min_size fallback shared by segment_target_objects
    and auto_segment_target: a physical size (from um) if given, else a fraction of
    the XY area clamped to [16, 512]. (segment_intensity_regions uses a different
    ``... or int(min_size)`` fallback and keeps it inline.)"""
    ndim = raw.ndim
    xy_area = int(np.prod(raw.shape[-2:])) if ndim >= 2 else int(raw.size)
    physical = min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=ndim,
    )
    return physical or max(16, min(512, int(round(xy_area * 0.00005))))


def finalize_qc_png(
    image,
    masks,
    labels_layer,
    source_layer,
    *,
    method,
    save_qc_png,
    qc_png_path,
    secondary_outline_mask=None,
):
    """The shared QC-PNG save block: resolve the output path (explicit
    ``qc_png_path`` else the default bundle path derived from the *source* layer),
    write the overlay, record it on the labels layer's metadata, and capture
    saved/skipped/error. Returns
    ``(saved_qc_png, qc_png_error, qc_png_skipped_reason)``. Needs both the labels
    layer object (``.name`` + the metadata write) and the source layer object
    (default path + provenance)."""
    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(labels_layer.name, source_layer)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                image,
                masks,
                out_path,
                labels_layer=labels_layer.name,
                source_layer=source_layer.name,
                method=method,
                force=qc_png_path is not None,
                secondary_outline_mask=secondary_outline_mask,
            )
            if saved_qc_png:
                try:
                    labels_layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"
    return saved_qc_png, qc_png_error, qc_png_skipped_reason
