"""Pure target-object segmentation pipeline (no napari, no viewer).

Extracted from ``tools.segment.segment_target_objects`` so the deterministic
auto-correct loop can iterate in memory and the single-shot tool, the loop, and
headless tests all share one code path (no drift). Split into a background step
and a threshold step (M2 in the ROI-judgment plan): background correction is the
expensive part and is invariant to the threshold parameters, so the loop runs it
once and re-thresholds per iteration.

The caller owns anything viewer-shaped: snapshotting the layer, resolving a
``boundary_mask`` layer to a bool array, saturation warnings, and adding the
output labels layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imajin.analysis.segmentation import (
    estimate_local_background,
    intersect_labels_with_mask,
    label_qc,
    labels_from_binary,
    target_object_qc,
)
from imajin.analysis.segmentation_auto3d import (
    confidence_from_score,
    roi_shape_metrics,
    score_roi_quality,
)
from imajin.analysis.target_segmentation import target_threshold_for_scope


@dataclass
class TargetSegmentation:
    """Result of one target-object segmentation pass."""

    masks: np.ndarray
    threshold: float
    high_threshold: float
    noise_sigma: float
    threshold_scope: str
    threshold_warnings: list[str]
    qc: dict[str, Any]
    signal_qc: dict[str, Any]
    qc_warnings: list[str]
    roi_score: float
    roi_confidence: str
    score_metrics: dict[str, Any] = field(default_factory=dict)


def prepare_corrected(
    raw: np.ndarray,
    *,
    background_radius: int,
    background_method: str,
    background_percentile: float,
    smoothing_sigma: float,
) -> np.ndarray:
    """Background-correct and optionally smooth the target image.

    This is the threshold-invariant, expensive step (M2): the auto-correct loop
    computes it once and re-thresholds in place, recomputing only when a
    background/smoothing parameter actually changes. Returns the corrected image
    used for thresholding and QC.
    """
    from skimage import filters

    raw = np.asarray(raw, dtype=np.float32)
    background = estimate_local_background(
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
        corrected = filters.gaussian(
            corrected,
            sigma=sigma,
            preserve_range=True,
        ).astype(np.float32)
    return corrected


def threshold_and_label(
    corrected_for_threshold: np.ndarray,
    raw: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    threshold_clip_percentile: float | None = None,
    auto_mask_hyperbright: bool = False,
    hyperbright_percentile: float = 99.5,
    hyperbright_dilate_radius: int = 2,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    boundary_mask: np.ndarray | None = None,
) -> TargetSegmentation:
    """Threshold an already-corrected image into labelled ROIs, with QC + score.

    Ports the threshold / hysteresis / labelling / QC / scoring block of
    ``segment_target_objects`` verbatim. ``boundary_mask`` is an optional bool
    array (caller resolves it from a layer). The returned ``qc_warnings`` are the
    target-object QC warnings only; the caller prepends saturation warnings and
    can prepend ``threshold_warnings``.
    """
    from skimage import filters

    raw = np.asarray(raw, dtype=np.float32)
    corrected_for_threshold = np.asarray(corrected_for_threshold, dtype=np.float32)

    threshold, noise_sigma, threshold_scope, threshold_warnings = target_threshold_for_scope(
        corrected_for_threshold,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        min_snr=min_snr,
        boundary_mask=boundary_mask,
        clip_percentile=threshold_clip_percentile,
        auto_mask_hyperbright=auto_mask_hyperbright,
        hyperbright_percentile=hyperbright_percentile,
        hyperbright_dilate_radius=hyperbright_dilate_radius,
    )

    high_threshold = max(float(threshold), float(high_snr) * float(noise_sigma))
    if boundary_mask is not None:
        scoped_threshold_image = np.where(
            boundary_mask,
            corrected_for_threshold,
            -np.inf,
        ).astype(np.float32, copy=False)
        low_candidates = (scoped_threshold_image >= float(threshold)) & boundary_mask
        high_seeds = (scoped_threshold_image >= high_threshold) & boundary_mask
        if high_threshold > threshold and np.any(high_seeds):
            binary = (
                filters.apply_hysteresis_threshold(
                    scoped_threshold_image,
                    low=float(threshold),
                    high=float(high_threshold),
                )
                & boundary_mask
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

    masks = labels_from_binary(
        binary,
        min_size=min_size,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )
    qc = label_qc(masks)
    signal_qc, qc_warnings = target_object_qc(
        raw,
        corrected_for_threshold,
        masks,
        noise_sigma=noise_sigma,
    )

    if boundary_mask is not None:
        masks = intersect_labels_with_mask(masks, boundary_mask, renumber=True)
        qc = label_qc(masks)
        signal_qc, qc_warnings = target_object_qc(
            raw,
            corrected_for_threshold,
            masks,
            noise_sigma=noise_sigma,
        )

    # Deterministic ROI-quality score on the *corrected* metrics (H2).
    score_metrics = {**qc, **signal_qc, **roi_shape_metrics(masks)}
    roi_score = score_roi_quality(
        score_metrics,
        [],
        noise_sigma=noise_sigma,
        ndim=masks.ndim,
        has_multiple_z=(masks.ndim == 3 and masks.shape[0] > 1),
    )
    roi_confidence = confidence_from_score(roi_score, score_metrics)

    return TargetSegmentation(
        masks=masks,
        threshold=float(threshold),
        high_threshold=float(high_threshold),
        noise_sigma=float(noise_sigma),
        threshold_scope=threshold_scope,
        threshold_warnings=list(threshold_warnings),
        qc=qc,
        signal_qc=signal_qc,
        qc_warnings=list(qc_warnings),
        roi_score=float(roi_score),
        roi_confidence=roi_confidence,
        score_metrics=score_metrics,
    )


def segment_target_array(
    raw: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    smoothing_sigma: float = 1.0,
    min_size: int,
    **threshold_kwargs: Any,
) -> TargetSegmentation:
    """Single-shot composition of :func:`prepare_corrected` +
    :func:`threshold_and_label`. Convenience for the tool and headless tests;
    the loop calls the two steps separately so it can reuse the corrected image.
    """
    corrected = prepare_corrected(
        raw,
        background_radius=background_radius,
        background_method=background_method,
        background_percentile=background_percentile,
        smoothing_sigma=smoothing_sigma,
    )
    return threshold_and_label(
        corrected,
        raw,
        spacing=spacing,
        min_size=min_size,
        **threshold_kwargs,
    )
