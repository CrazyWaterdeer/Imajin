from __future__ import annotations

import numpy as np

from imajin.analysis.segmentation import (
    robust_background_sigma,
    target_object_threshold,
)


def target_threshold_for_scope(
    corrected: np.ndarray,
    *,
    threshold_method: str,
    threshold_percentile: float,
    min_snr: float,
    boundary_mask: np.ndarray | None = None,
) -> tuple[float, float, str, list[str]]:
    full_noise_sigma = robust_background_sigma(corrected)
    warnings: list[str] = []
    if boundary_mask is None:
        return (
            target_object_threshold(
                corrected,
                method=threshold_method,
                percentile=threshold_percentile,
                min_snr=min_snr,
                noise_sigma=full_noise_sigma,
            ),
            full_noise_sigma,
            "full_image",
            warnings,
        )

    scoped_mask = np.asarray(boundary_mask, dtype=bool) & np.isfinite(corrected)
    if not np.any(scoped_mask):
        warnings.append(
            "boundary mask contains no finite target pixels; full-image threshold "
            "was used before mask intersection"
        )
        return (
            target_object_threshold(
                corrected,
                method=threshold_method,
                percentile=threshold_percentile,
                min_snr=min_snr,
                noise_sigma=full_noise_sigma,
            ),
            full_noise_sigma,
            "full_image_fallback",
            warnings,
        )

    scoped_values = np.asarray(corrected[scoped_mask], dtype=np.float32)
    if float(np.max(scoped_values)) <= float(np.min(scoped_values)):
        warnings.append(
            "target intensities inside the boundary mask were constant; full-image "
            "threshold was used before mask intersection"
        )
        return (
            target_object_threshold(
                corrected,
                method=threshold_method,
                percentile=threshold_percentile,
                min_snr=min_snr,
                noise_sigma=full_noise_sigma,
            ),
            full_noise_sigma,
            "full_image_fallback",
            warnings,
        )

    scoped_noise_sigma = robust_background_sigma(scoped_values)
    return (
        target_object_threshold(
            scoped_values,
            method=threshold_method,
            percentile=threshold_percentile,
            min_snr=min_snr,
            noise_sigma=scoped_noise_sigma,
        ),
        scoped_noise_sigma,
        "boundary_mask",
        warnings,
    )
