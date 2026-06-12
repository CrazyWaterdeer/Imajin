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

from imajin.analysis.segmentation_auto3d import confidence_from_score

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

# --- v2.1 distribution-layer tunables (placeholders; calibrated by the F1 harness) ---
# All thresholds operate on log2(size). A merged doublet is +1.0 log2 (2x); a
# fragment tail sits well below the median.
DIST_SMALL_TAIL_LOG2 = 1.0  # "small" = this many log2 units below the median
DIST_SMALL_TAIL_FRACTION = 0.20  # fraction below that cutoff to flag over-segmentation
DIST_BIMODAL_LOG2_GAP = 0.6  # interior log2 gap (with both sides substantial) = bimodal
# F1 regression floors: the validation harness fails below these.
VALIDATION_MIN_SENSITIVITY = 0.7
VALIDATION_MIN_SPECIFICITY = 0.7


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

    # Usable = interior (non-border) AND actually present (relabeling can leave
    # empty label ids with zero voxels; those are not objects).
    n_usable = int(np.count_nonzero((~border_mask) & (sizes > 0)))
    return sizes, border_mask, n_usable


def distribution_flag(sizes_um: np.ndarray, *, n_eff: int) -> dict[str, Any]:
    """Layer 2 — a weak, biology-safe size-distribution anomaly flag.

    Operates on log2(size) with robust stats. Returns only a flag (with the
    offending metric); it is **never** a score delta and **never** yields
    ``low``/``high`` — at most it routes to ``medium`` ("worth a look"). High
    spread that is merely *broad and unimodal* (real biology) is not flagged;
    only a fragment tail (over-segmentation) or a distinct ~2× secondary mode
    (under-segmentation) is. Below ``MIN_DISTRIBUTION_N`` effective objects the
    tests are low-power, so it **abstains** rather than asserting coherence.
    """
    sizes = np.asarray(sizes_um, dtype=float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if int(n_eff) < MIN_DISTRIBUTION_N or sizes.size < MIN_DISTRIBUTION_N:
        return {"flag": False, "reason": None, "metric": None, "abstained": True}

    log2 = np.sort(np.log2(sizes))
    med = float(np.median(log2))

    # Over-segmentation: a substantial small-size tail.
    small_frac = float(np.mean(log2 < med - DIST_SMALL_TAIL_LOG2))
    if small_frac >= DIST_SMALL_TAIL_FRACTION:
        return {
            "flag": True,
            "reason": "possible_oversegmentation",
            "metric": round(small_frac, 3),
            "abstained": False,
        }

    # Under-segmentation: a distinct secondary mode (largest interior gap with
    # both sides substantial) — merged doublets sit ~+1.0 log2 (2×).
    gaps = np.diff(log2)
    if gaps.size:
        k = int(np.argmax(gaps))
        gap = float(gaps[k])
        frac_low = (k + 1) / log2.size
        if gap >= DIST_BIMODAL_LOG2_GAP and min(frac_low, 1.0 - frac_low) >= DIST_SMALL_TAIL_FRACTION:
            return {
                "flag": True,
                "reason": "possible_undersegmentation",
                "metric": round(gap, 3),
                "abstained": False,
            }

    return {"flag": False, "reason": None, "metric": None, "abstained": False}


def correction_materiality(
    raw_qc: dict[str, Any],
    corrected_qc: dict[str, Any],
    *,
    count_tol: float = 0.25,
    size_tol: float = 0.5,
) -> bool:
    """True when the auto-correct loop moved the measurement materially.

    Compares object count and median object size between the raw first pass and
    the corrected mask; a material gap means the distribution layer must not
    bless the corrected mask as "coherent" (it may be a correction artifact).
    """
    rn = int(raw_qc.get("n_objects", 0) or 0)
    cn = int(corrected_qc.get("n_objects", 0) or 0)
    if rn == 0:
        return cn > 0
    if abs(cn - rn) / rn > count_tol:
        return True
    rmed = float(raw_qc.get("object_area_median", 0.0) or 0.0)
    cmed = float(corrected_qc.get("object_area_median", 0.0) or 0.0)
    if rmed > 0 and abs(cmed - rmed) / rmed > size_tol:
        return True
    return False


def roi_confidence_v2(
    structural_score: float,
    structural_metrics: dict[str, Any],
    *,
    route_layers: set[str],
    n_eff: int,
    obj_class: ObjectClass,
    dist_flag: dict[str, Any] | None,
    correction_gap: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Evidence-based confidence ("high"/"medium"/"low") + the drivers behind it.

    - `low` only on gross structural failure (Layer 1).
    - `high` requires *positive* evidence: strong structural separation AND the
      distribution layer was applicable, checked, and silent — so it is reserved
      for numerous, distributed blobs. Domains / sparse fields / neurons (no
      distribution layer) are capped at `medium`, closing v1's "absence of
      penalty = high" hole.
    - `medium` (→ show image) for a distribution flag, a material correction gap,
      or simply insufficient positive evidence.
    """
    drivers: dict[str, Any] = {
        "object_class": obj_class,
        "structural_score": round(float(structural_score), 1),
        "n_eff": int(n_eff),
    }

    struct_tier = confidence_from_score(structural_score, structural_metrics)
    if struct_tier == "low":
        drivers["driver"] = "structural_failure"
        return "low", drivers

    if dist_flag and dist_flag.get("flag"):
        drivers["driver"] = dist_flag.get("reason") or "distribution_anomaly"
        drivers["distribution_metric"] = dist_flag.get("metric")
        return "medium", drivers

    if correction_gap:
        drivers["driver"] = "correction_changed_measurement"
        return "medium", drivers

    distribution_checked = (
        "distribution" in route_layers
        and dist_flag is not None
        and not dist_flag.get("abstained", False)
    )
    if struct_tier == "high" and distribution_checked:
        drivers["driver"] = "structural_strong_and_distribution_clean"
        return "high", drivers

    drivers["driver"] = "insufficient_positive_evidence"
    return "medium", drivers


def assess_roi(
    masks: np.ndarray,
    spacing: tuple[float, ...] | None,
    structural_score: float,
    structural_metrics: dict[str, Any],
    *,
    obj_class: ObjectClass = "blob",
    raw_qc: dict[str, Any] | None = None,
    corrected_qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end v2.1 verdict for one labelled mask.

    Composes Layer 0 routing → size extraction → the distribution flag →
    `roi_confidence_v2`, plus correction materiality when raw/corrected QC are
    given. The single integration point so both tools (and any caller) stay
    consistent. Returns `roi_confidence`, `distribution_flag`,
    `confidence_drivers`, `correction_gap`.
    """
    n_eff, distributed = effective_object_count(masks)
    layers = route(obj_class, n_eff, distributed)

    dist_flag: dict[str, Any] | None = None
    if "distribution" in layers:
        sizes, border_mask, n_usable = object_sizes_physical(masks, spacing)
        dist_flag = distribution_flag(sizes[~border_mask], n_eff=n_usable)

    correction_gap = bool(
        raw_qc is not None
        and corrected_qc is not None
        and correction_materiality(raw_qc, corrected_qc)
    )

    confidence, drivers = roi_confidence_v2(
        structural_score,
        structural_metrics,
        route_layers=layers,
        n_eff=n_eff,
        obj_class=obj_class,
        dist_flag=dist_flag,
        correction_gap=correction_gap,
    )
    return {
        "roi_confidence": confidence,
        "distribution_flag": dist_flag,
        "confidence_drivers": drivers,
        "correction_gap": correction_gap,
    }
