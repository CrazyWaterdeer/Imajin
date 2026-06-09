from __future__ import annotations

import numpy as np

from imajin.analysis.segmentation import (
    robust_background_sigma,
    target_object_threshold,
)


def _hyperbright_exclusion_mask(
    corrected: np.ndarray,
    *,
    percentile: float,
    dilate_radius: int,
) -> np.ndarray:
    """Pixels at or above `percentile` of the finite values, dilated in XY.

    Returned as a boolean mask the same shape as `corrected`. Non-finite
    pixels are treated as hyper-bright (excluded) so they cannot leak into
    threshold statistics.
    """
    from scipy import ndimage as ndi

    finite_mask = np.isfinite(corrected)
    finite = np.asarray(corrected[finite_mask], dtype=np.float32)
    if finite.size == 0:
        return np.ones(corrected.shape, dtype=bool)

    cap = float(np.percentile(finite, float(percentile)))
    if not np.isfinite(cap):
        return ~finite_mask

    hyper = (~finite_mask) | (corrected >= cap)
    if dilate_radius <= 0:
        return hyper

    if corrected.ndim == 3:
        structure_shape = (1, 2 * dilate_radius + 1, 2 * dilate_radius + 1)
    else:
        structure_shape = tuple(2 * dilate_radius + 1 for _ in range(corrected.ndim))
    structure = np.ones(structure_shape, dtype=bool)
    return ndi.binary_dilation(hyper, structure=structure)


def target_threshold_for_scope(
    corrected: np.ndarray,
    *,
    threshold_method: str,
    threshold_percentile: float,
    min_snr: float,
    boundary_mask: np.ndarray | None = None,
    clip_percentile: float | None = None,
    auto_mask_hyperbright: bool = False,
    hyperbright_percentile: float = 99.5,
    hyperbright_dilate_radius: int = 2,
) -> tuple[float, float, str, list[str]]:
    warnings: list[str] = []

    # Optionally exclude hyper-bright outliers from the threshold scope so
    # autofluorescence/debris cannot drag the histogram or robust sigma upward.
    # The exclusion is intersected with any caller-supplied boundary mask.
    effective_boundary: np.ndarray | None
    if boundary_mask is None:
        effective_boundary = None
    else:
        effective_boundary = np.asarray(boundary_mask, dtype=bool)

    auto_mask_used = False
    if auto_mask_hyperbright:
        exclusion = _hyperbright_exclusion_mask(
            corrected,
            percentile=float(hyperbright_percentile),
            dilate_radius=int(hyperbright_dilate_radius),
        )
        include = ~exclusion & np.isfinite(corrected)
        if np.any(include):
            effective_boundary = (
                include
                if effective_boundary is None
                else (effective_boundary & include)
            )
            auto_mask_used = True
        else:
            warnings.append(
                "auto_mask_hyperbright excluded every finite pixel; falling "
                "back to the unmasked threshold scope"
            )

    full_noise_sigma = robust_background_sigma(corrected)

    if effective_boundary is None:
        return (
            target_object_threshold(
                corrected,
                method=threshold_method,
                percentile=threshold_percentile,
                min_snr=min_snr,
                noise_sigma=full_noise_sigma,
                clip_percentile=clip_percentile,
            ),
            full_noise_sigma,
            "full_image",
            warnings,
        )

    scoped_mask = effective_boundary & np.isfinite(corrected)
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
                clip_percentile=clip_percentile,
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
                clip_percentile=clip_percentile,
            ),
            full_noise_sigma,
            "full_image_fallback",
            warnings,
        )

    scoped_noise_sigma = robust_background_sigma(scoped_values)
    if auto_mask_used and boundary_mask is None:
        scope_label = "hyperbright_auto_masked"
    elif auto_mask_used:
        scope_label = "boundary_mask_hyperbright_masked"
    else:
        scope_label = "boundary_mask"
    return (
        target_object_threshold(
            scoped_values,
            method=threshold_method,
            percentile=threshold_percentile,
            min_snr=min_snr,
            noise_sigma=scoped_noise_sigma,
            clip_percentile=clip_percentile,
        ),
        scoped_noise_sigma,
        scope_label,
        warnings,
    )
