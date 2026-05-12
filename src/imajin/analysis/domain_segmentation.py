from __future__ import annotations

from typing import Any

import numpy as np

from imajin.analysis.segmentation import min_size_from_physical


def domain_smoothing_sigma(
    spacing: tuple[float, ...] | None,
    ndim: int,
    smooth_sigma_um: float,
) -> float | tuple[float, ...]:
    sigma = float(smooth_sigma_um)
    if sigma <= 0:
        return 0.0
    if spacing is None:
        if ndim == 3:
            return (0.0, sigma, sigma)
        return sigma
    if ndim == 3:
        return (
            0.0,
            sigma / float(spacing[1]),
            sigma / float(spacing[2]),
        )
    if ndim == 2:
        return (
            sigma / float(spacing[0]),
            sigma / float(spacing[1]),
        )
    return tuple(sigma / float(s) for s in spacing[:ndim])


def smooth_domain_image(
    raw: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    smooth_sigma_um: float,
) -> np.ndarray:
    if smooth_sigma_um <= 0:
        return raw
    from scipy import ndimage as ndi

    finite = np.isfinite(raw)
    if np.any(finite):
        fill = float(np.median(raw[finite]))
    else:
        fill = 0.0
    work = np.where(finite, raw, fill).astype(np.float32, copy=False)
    sigma = domain_smoothing_sigma(spacing, raw.ndim, smooth_sigma_um)
    return ndi.gaussian_filter(work, sigma=sigma).astype(np.float32, copy=False)


def domain_min_size_from_physical(
    *,
    min_area_um2: float | None,
    min_volume_um3: float | None,
    spacing: tuple[float, ...] | None,
    ndim: int,
) -> int | None:
    area = (
        float(min_area_um2)
        if min_area_um2 is not None and float(min_area_um2) > 0
        else None
    )
    volume = (
        float(min_volume_um3)
        if min_volume_um3 is not None and float(min_volume_um3) > 0
        else None
    )
    if area is None and volume is None:
        return None
    if spacing is None:
        return None
    if ndim == 3 and volume is None and area is not None:
        xy_area = float(spacing[1] * spacing[2])
        if xy_area > 0:
            return max(1, int(np.ceil(area / xy_area)))
    return min_size_from_physical(
        min_size=None,
        min_volume_um3=volume,
        min_area_um2=area,
        spacing=spacing,
        ndim=ndim,
    )


def domain_physical_sizes(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None,
) -> dict[str, float | int | None]:
    voxels = int(np.count_nonzero(mask))
    if spacing is None:
        return {
            "domain_voxels": voxels,
            "domain_area_um2": float(voxels),
            "domain_volume_um3": None,
        }
    if mask.ndim == 3:
        xy_area = float(spacing[1] * spacing[2])
        volume = float(spacing[0] * spacing[1] * spacing[2])
        return {
            "domain_voxels": voxels,
            "domain_area_um2": float(voxels) * xy_area,
            "domain_volume_um3": float(voxels) * volume,
        }
    area = float(spacing[0] * spacing[1])
    return {
        "domain_voxels": voxels,
        "domain_area_um2": float(voxels) * area,
        "domain_volume_um3": None,
    }


def filter_domain_components(
    binary: np.ndarray,
    *,
    max_components: int | None,
    min_component_fraction: float,
    merge_components: bool,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    from skimage import measure, segmentation

    labels_raw = measure.label(np.asarray(binary, dtype=bool), connectivity=1).astype(
        np.int32
    )
    n_raw = int(labels_raw.max()) if labels_raw.size else 0
    if n_raw == 0:
        return (
            np.zeros(labels_raw.shape, dtype=np.int32),
            {
                "n_components_raw": 0,
                "n_components_retained": 0,
                "domain_label_count": 0,
            },
            [],
        )

    counts = np.bincount(labels_raw.ravel(), minlength=n_raw + 1)
    keep = np.zeros(n_raw + 1, dtype=bool)
    keep[1:] = True
    foreground = int(counts[1:].sum())
    min_fraction = max(0.0, float(min_component_fraction))
    if min_fraction > 0 and foreground > 0:
        min_voxels = max(1, int(np.ceil(foreground * min_fraction)))
        keep &= counts >= min_voxels
        keep[0] = False

    kept_ids = np.flatnonzero(keep)
    capped = False
    if max_components is not None and int(max_components) > 0:
        limit = int(max_components)
        if kept_ids.size > limit:
            order = np.argsort(counts[kept_ids])[::-1]
            kept_ids = kept_ids[order[:limit]]
            capped = True
            keep[:] = False
            keep[kept_ids] = True

    filtered = np.where(keep[labels_raw], labels_raw, 0).astype(np.int32, copy=False)
    retained = int(kept_ids.size)
    if retained == 0:
        labels = np.zeros(labels_raw.shape, dtype=np.int32)
    elif merge_components:
        labels = (filtered > 0).astype(np.int32)
    else:
        labels, _fw, _inv = segmentation.relabel_sequential(filtered)
        labels = np.asarray(labels, dtype=np.int32)

    warnings: list[str] = []
    if capped:
        warnings.append(
            f"expression domain had {n_raw} connected components after cleanup; "
            f"kept the largest {retained} components. Increase min_area_um2, "
            "min_volume_um3, or k_mad if this still reflects noise"
        )
    elif n_raw > 1000:
        warnings.append(
            f"expression domain still has {n_raw} connected components after "
            "cleanup; inspect the QC image and consider increasing min_area_um2 "
            "or k_mad"
        )

    return (
        labels,
        {
            "n_components_raw": n_raw,
            "n_components_retained": retained,
            "domain_label_count": int(labels.max()) if labels.size else 0,
        },
        warnings,
    )
