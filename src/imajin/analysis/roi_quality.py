"""Context-aware, evidence-based ROI confidence (scorer v2.1).

A leaf module: it builds on the structural scorer (`score_roi_quality` in
`segmentation_auto3d`) but is never imported *by* auto3d, so the existing
auto3d → target_segmentation import chain is untouched.

This file grows over the v2.1 plan. Phase A lands Layer 0 — routing by object
class and effective object count — which is a hard prerequisite: every later
layer keys off the route. Mis-routing poisons the whole scorer, so unknown
inputs stay conservative (structural + vision, never the distribution layer).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Object classes the scorer routes on.
ObjectClass = str  # "blob" | "domain" | "neuron" | "unclassified"

_BLOB_METHODS = {
    "target_objects",
    "auto_target_objects",
    "auto_3d_cells",
    "cellpose_sam",
    "intensity_regions",
}
_DOMAIN_METHODS = {"expression_domain"}
_NEURON_HINTS = ("neuron", "skeleton", "trace", "process")

# Eligibility for the distribution layer (Layer 2). Centralized in the F0
# constants step later; kept here so Phase A is self-contained.
MIN_DISTRIBUTION_N = 10
_MIN_SPATIAL_SPREAD = 0.25  # fraction of each in-plane axis the centroids must span


def object_class(meta: dict[str, Any] | None) -> ObjectClass:
    """Classify a labels layer from its metadata.

    Reads `segmentation_method` first, then falls back to `object_unit`.
    Anything unrecognized → "unclassified" (routed conservatively).
    """
    m = meta or {}
    method = str(m.get("segmentation_method", "")).strip().lower()
    if method in _BLOB_METHODS:
        return "blob"
    if method in _DOMAIN_METHODS:
        return "domain"
    if any(h in method for h in _NEURON_HINTS):
        return "neuron"

    unit = str(m.get("object_unit", "")).strip().lower()
    if unit in {"object_or_roi", "cell", "nucleus", "punctum", "puncta", "droplet"}:
        return "blob"
    if "domain" in unit:
        return "domain"
    if any(h in unit for h in _NEURON_HINTS):
        return "neuron"
    return "unclassified"


def effective_object_count(labels: np.ndarray) -> tuple[int, bool]:
    """Return (n_objects, spatially_distributed).

    `spatially_distributed` is True when object centroids span at least
    `_MIN_SPATIAL_SPREAD` of each in-plane (Y, X) axis — so N objects crammed
    into one corner of a crop are not treated as N independent examples
    (Codex). Needs >= 2 objects to be meaningful.
    """
    labels = np.asarray(labels)
    n = int(labels.max()) if labels.size else 0
    if n < 2:
        return n, False

    from scipy import ndimage as ndi

    coms = ndi.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8), labels, index=list(range(1, n + 1))
    )
    coms = np.asarray(coms, dtype=float)
    if coms.ndim == 1:
        coms = coms[None, :]
    yx = coms[:, -2:]
    shape_yx = np.asarray(labels.shape[-2:], dtype=float)
    span = (yx.max(axis=0) - yx.min(axis=0)) / np.maximum(shape_yx, 1.0)
    distributed = bool(np.all(span >= _MIN_SPATIAL_SPREAD))
    return n, distributed


def route(
    obj_class: ObjectClass,
    n_eff: int,
    distributed: bool,
    *,
    min_distribution_n: int = MIN_DISTRIBUTION_N,
) -> set[str]:
    """Which layers apply: always structural + vision; distribution only for
    blob-like objects that are numerous AND spatially distributed.

    Domain (one region), neuron/arbor (no meaningful size distribution; no
    morphology-consistency layer in v2.1), and unclassified all skip the
    distribution layer.
    """
    layers = {"structural", "vision"}
    if obj_class == "blob" and n_eff >= int(min_distribution_n) and distributed:
        layers.add("distribution")
    return layers


def _labels_touching_border(labels: np.ndarray) -> set[int]:
    """Label ids present on any face of the array (truncated / cropped objects)."""
    border: set[int] = set()
    for ax in range(labels.ndim):
        border |= set(np.take(labels, 0, axis=ax).ravel().tolist())
        border |= set(np.take(labels, -1, axis=ax).ravel().tolist())
    border.discard(0)
    return border


def object_sizes_physical(
    labels: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-object physical size + a border-exclusion mask.

    Returns ``(sizes, border_mask, n_usable)`` where ``sizes[i]`` is the
    physical size of label ``i+1`` — **area in µm² for 2D, volume in µm³ for
    3D** (anisotropy via ``spacing``; pixel units when spacing is None) — and
    ``border_mask[i]`` is True when that label touches an array face. Truncated
    border objects must be excluded by the caller (``sizes[~border_mask]``)
    because truncation inflates the small-size tail and fakes multimodality.
    Never mix 2D and 3D sizes in one vector.
    """
    labels = np.asarray(labels)
    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool), 0

    if spacing is None:
        voxel = 1.0
    else:
        voxel = float(np.prod(np.asarray(spacing, dtype=float)))

    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    sizes = counts.astype(float) * voxel

    border_mask = np.zeros(n, dtype=bool)
    for lab in _labels_touching_border(labels):
        if 1 <= lab <= n:
            border_mask[lab - 1] = True

    n_usable = int(np.count_nonzero(~border_mask))
    return sizes, border_mask, n_usable
