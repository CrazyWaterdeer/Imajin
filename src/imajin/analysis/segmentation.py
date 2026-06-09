from __future__ import annotations

from typing import Any

import numpy as np


def label_qc(masks: np.ndarray) -> dict[str, Any]:
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
    counts = np.bincount(masks.ravel(), minlength=n + 1)[1:]
    qc["object_area_min"] = int(counts.min())
    qc["object_area_median"] = float(np.median(counts))
    qc["object_area_max"] = int(counts.max())
    return qc


def label_qc_warnings(masks: np.ndarray) -> list[str]:
    if masks.size == 0:
        return ["empty label image"]
    n = int(masks.max())
    if n == 0:
        return ["segmentation produced zero objects"]
    counts = np.bincount(masks.ravel(), minlength=n + 1)[1:]
    median = float(np.median(counts))
    largest = int(counts.max())
    image_area = int(masks.size)
    warnings: list[str] = []
    if n < 3:
        warnings.append(
            "very few objects found; this may be region-level ROI segmentation, "
            "not cell-level segmentation"
        )
    if image_area > 0 and largest / image_area > 0.05:
        warnings.append(
            "largest object covers more than 5% of the XY field; check for merged "
            "regions or wrong target channel"
        )
    if median > 0 and largest / median > 8:
        warnings.append(
            "object sizes are highly uneven; segmentation may include merged "
            "regions or debris"
        )
    return warnings


def voxel_spacing(scale: tuple[float, ...] | None, ndim: int) -> tuple[float, ...] | None:
    if not scale:
        return None
    spacing = tuple(float(s) for s in scale[:ndim])
    if len(spacing) != ndim:
        return None
    if any(s <= 0 or not np.isfinite(s) for s in spacing):
        return None
    return spacing


def min_size_from_physical(
    *,
    min_size: int | None,
    min_volume_um3: float | None,
    min_area_um2: float | None,
    spacing: tuple[float, ...] | None,
    ndim: int,
) -> int | None:
    if spacing is None:
        return int(min_size) if min_size is not None else None
    if ndim == 3 and min_volume_um3 is not None:
        voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
        if voxel_volume > 0:
            return max(1, int(np.ceil(float(min_volume_um3) / voxel_volume)))
    if ndim == 2 and min_area_um2 is not None:
        pixel_area = float(spacing[0] * spacing[1])
        if pixel_area > 0:
            return max(1, int(np.ceil(float(min_area_um2) / pixel_area)))
    if min_size is not None:
        return int(min_size)
    return None


def threshold_value(
    image: np.ndarray,
    method: str,
    percentile: float,
) -> float:
    from skimage import filters

    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("cannot threshold an image with no finite pixels")
    if float(finite.max()) <= float(finite.min()):
        raise ValueError("cannot threshold a constant image")

    key = method.lower().strip()
    if key == "percentile":
        return float(np.percentile(finite, percentile))
    if key == "otsu":
        return float(filters.threshold_otsu(finite))
    if key == "yen":
        return float(filters.threshold_yen(finite))
    if key == "li":
        return float(filters.threshold_li(finite))
    if key == "triangle":
        return float(filters.threshold_triangle(finite))
    raise ValueError(
        "threshold_method must be one of: percentile, otsu, yen, li, triangle"
    )


def remove_small_binary_objects(binary: np.ndarray, min_size: int) -> np.ndarray:
    import inspect

    from skimage import morphology

    threshold = max(1, int(min_size))
    params = inspect.signature(morphology.remove_small_objects).parameters
    if "max_size" in params:
        return morphology.remove_small_objects(binary, max_size=max(0, threshold - 1))
    return morphology.remove_small_objects(binary, min_size=threshold)


def remove_small_binary_holes(binary: np.ndarray, min_size: int) -> np.ndarray:
    import inspect

    from skimage import morphology

    threshold = max(1, int(min_size))
    params = inspect.signature(morphology.remove_small_holes).parameters
    if "max_size" in params:
        return morphology.remove_small_holes(binary, max_size=max(0, threshold - 1))
    return morphology.remove_small_holes(binary, area_threshold=threshold)


def remove_small_labeled_objects(labels: np.ndarray, min_size: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return arr
    counts = np.bincount(arr.ravel(), minlength=n + 1)
    keep = counts >= max(1, int(min_size))
    keep[0] = False
    return np.where(keep[arr], arr, 0).astype(np.int32)


def intersect_labels_with_mask(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    renumber: bool = False,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    binary = np.asarray(mask, dtype=bool)
    if arr.shape != binary.shape:
        raise ValueError(
            f"labels shape {arr.shape} does not match mask shape {binary.shape}"
        )
    out = np.where(binary, arr, 0).astype(np.int32, copy=False)
    if not renumber:
        return out
    unique = np.unique(out)
    unique = unique[unique > 0]
    if unique.size == 0:
        return out
    remap = np.zeros(int(unique.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[out]


def dilate_binary_um(
    binary: np.ndarray,
    *,
    spacing: tuple[float, ...],
    radius_um: float,
) -> np.ndarray:
    from scipy import ndimage as ndi

    if radius_um <= 0:
        return binary
    pixel_radius_per_axis: list[int] = []
    for sp in spacing[-binary.ndim:]:
        pr = max(1, int(round(float(radius_um) / float(sp))))
        pixel_radius_per_axis.append(pr)
    structure = np.ones(
        tuple(2 * r + 1 for r in pixel_radius_per_axis), dtype=bool
    )
    return ndi.binary_dilation(binary, structure=structure)


def estimate_local_background(
    image: np.ndarray,
    *,
    radius: int,
    method: str,
    percentile: float,
) -> np.ndarray:
    from scipy import ndimage as ndi

    data = np.asarray(image, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)

    if radius <= 0:
        return np.full(data.shape, float(np.percentile(finite, percentile)), np.float32)

    size = _xy_filter_size(data.ndim, int(radius))
    key = method.lower().strip().replace("-", "_")
    if key in {"opening", "morphology", "morphological_opening"}:
        return ndi.grey_opening(data, size=size, mode="nearest").astype(np.float32)
    if key in {"percentile", "local_percentile"}:
        return ndi.percentile_filter(
            data,
            percentile=float(percentile),
            size=size,
            mode="nearest",
        ).astype(np.float32)
    raise ValueError("background_method must be opening or percentile")


def robust_background_sigma(corrected: np.ndarray) -> float:
    finite = np.asarray(corrected[np.isfinite(corrected)], dtype=np.float32)
    if finite.size == 0:
        return 0.0
    cutoff = float(np.percentile(finite, 70))
    bg = finite[finite <= cutoff]
    if bg.size == 0:
        bg = finite
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    sigma = 1.4826 * mad
    if np.isfinite(sigma) and sigma > 0:
        return float(sigma)
    p50, p84 = np.percentile(bg, (50, 84))
    sigma = float((p84 - p50) / 0.994)
    if np.isfinite(sigma) and sigma > 0:
        return sigma
    return 0.0


def threshold_noise_floor(
    image: np.ndarray,
    *,
    k_mad: float,
    dark_percentile: float,
) -> float:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0
    if float(finite.max()) <= float(finite.min()):
        return float(finite.min())
    cutoff = float(np.percentile(finite, dark_percentile))
    dark = finite[finite <= cutoff]
    if dark.size == 0:
        dark = finite
    med = float(np.median(dark))
    mad = float(np.median(np.abs(dark - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        return med
    return med + float(k_mad) * sigma


def target_object_threshold(
    corrected: np.ndarray,
    *,
    method: str,
    percentile: float,
    min_snr: float,
    noise_sigma: float,
    clip_percentile: float | None = None,
) -> float:
    from skimage import filters

    finite = np.asarray(corrected[np.isfinite(corrected)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("cannot threshold an image with no finite pixels")
    if float(finite.max()) <= float(finite.min()):
        raise ValueError("cannot threshold a constant image")

    # Hyper-bright outliers (autofluorescence, debris) pull histogram-based
    # thresholds upward and starve dim signal. Optionally clip the upper tail
    # before passing the histogram to skimage's threshold algorithms. The
    # threshold value itself stays in the original intensity space.
    histogram_input = finite
    if clip_percentile is not None and 0.0 < float(clip_percentile) < 100.0:
        cap = float(np.percentile(finite, float(clip_percentile)))
        if np.isfinite(cap) and cap > float(finite.min()):
            histogram_input = np.minimum(finite, cap)

    key = method.lower().strip().replace("-", "_")
    if key in {"auto", "background_corrected", "target"}:
        try:
            threshold = float(filters.threshold_otsu(histogram_input))
        except ValueError:
            positives = finite[finite > 0]
            threshold = float(np.percentile(positives, 25)) if positives.size else 0.0
    elif key == "percentile":
        threshold = float(np.percentile(finite, percentile))
    elif key == "otsu":
        threshold = float(filters.threshold_otsu(histogram_input))
    elif key == "yen":
        threshold = float(filters.threshold_yen(histogram_input))
    elif key == "li":
        threshold = float(filters.threshold_li(histogram_input))
    elif key == "triangle":
        threshold = float(filters.threshold_triangle(histogram_input))
    elif key in {"multi_otsu", "multiotsu"}:
        # 3-class Multi-Otsu: background / signal / hyper-bright. The lower
        # boundary separates background from signal, which is what we want.
        try:
            boundaries = filters.threshold_multiotsu(histogram_input, classes=3)
            threshold = float(boundaries[0])
        except ValueError:
            threshold = float(filters.threshold_otsu(histogram_input))
    else:
        raise ValueError(
            "threshold_method must be one of: auto, percentile, otsu, yen, li, "
            "triangle, multi_otsu"
        )

    snr_floor = float(min_snr) * float(noise_sigma)
    if np.isfinite(snr_floor) and snr_floor > 0:
        threshold = max(threshold, snr_floor)
    return float(threshold)


def target_object_qc(
    image: np.ndarray,
    corrected: np.ndarray,
    masks: np.ndarray,
    *,
    noise_sigma: float,
) -> tuple[dict[str, Any], list[str]]:
    labels = np.asarray(masks)
    mask = labels > 0
    finite = np.isfinite(corrected)
    if not np.any(finite):
        return {}, ["target object QC could not be computed: no finite pixels"]

    values = corrected[finite]
    mask_finite = mask & finite
    outside = (~mask) & finite
    qc: dict[str, Any] = {
        "mask_fraction": float(mask_finite.sum() / max(1, int(finite.sum()))),
        "noise_sigma": float(noise_sigma),
    }
    warnings = label_qc_warnings(labels)

    if np.any(mask_finite):
        inside_values = corrected[mask_finite]
        qc["inside_corrected_mean"] = float(np.mean(inside_values))
        qc["inside_raw_mean"] = float(np.mean(image[mask_finite]))
    else:
        qc["inside_corrected_mean"] = 0.0
        qc["inside_raw_mean"] = 0.0

    if np.any(outside):
        outside_values = corrected[outside]
        qc["outside_corrected_mean"] = float(np.mean(outside_values))
        qc["outside_raw_mean"] = float(np.mean(image[outside]))
    else:
        qc["outside_corrected_mean"] = 0.0
        qc["outside_raw_mean"] = 0.0

    bright_threshold = float(np.percentile(values, 99))
    bright = finite & (corrected >= bright_threshold)
    bright_total = int(bright.sum())
    if bright_total > 0:
        outside_bright = int((bright & ~mask).sum())
        qc["top_bright_outside_fraction"] = float(outside_bright / bright_total)
        if qc["top_bright_outside_fraction"] > 0.25:
            warnings.append(
                "many top-bright pixels are outside the labels; target signal may be "
                "missed"
            )
    else:
        qc["top_bright_outside_fraction"] = 0.0

    if qc["mask_fraction"] > 0.5:
        warnings.append(
            "labels cover more than half of the image; background may be included"
        )
    separation = qc["inside_corrected_mean"] - qc["outside_corrected_mean"]
    if noise_sigma > 0 and separation < noise_sigma:
        warnings.append(
            "inside/outside corrected intensity separation is weak; target/background "
            "distinction is uncertain"
        )
    return qc, warnings


def segment_connected_regions(
    image: np.ndarray,
    *,
    threshold_method: str,
    percentile: float,
    min_size: int,
    smoothing_sigma: float,
    fill_holes: bool,
    split_touching: bool,
    min_distance: int,
    min_distance_um: float | None = None,
    spacing: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, float]:
    from scipy import ndimage as ndi
    from skimage import filters, measure, segmentation
    from skimage.feature import peak_local_max

    data = np.asarray(image, dtype=np.float32)
    if smoothing_sigma > 0:
        sigma: float | tuple[float, ...]
        if data.ndim == 3:
            # Smooth within XY planes; z spacing is usually much coarser in confocal data.
            sigma = (0.0, float(smoothing_sigma), float(smoothing_sigma))
        else:
            sigma = float(smoothing_sigma)
        data = filters.gaussian(data, sigma=sigma, preserve_range=True).astype(np.float32)

    threshold = threshold_value(data, threshold_method, percentile)
    binary = np.isfinite(data) & (data > threshold)
    binary = remove_small_binary_objects(binary, min_size)
    if fill_holes:
        binary = remove_small_binary_holes(binary, min_size)

    if not np.any(binary):
        return np.zeros(data.shape, dtype=np.int32), threshold

    if split_touching:
        distance = ndi.distance_transform_edt(binary, sampling=spacing)
        footprint = _physical_peak_footprint(spacing, min_distance_um, binary.ndim)
        peak_kwargs = {
            "labels": binary,
            "exclude_border": False,
        }
        if footprint is not None:
            peak_kwargs["footprint"] = footprint
            peak_kwargs["min_distance"] = 1
        else:
            peak_kwargs["min_distance"] = max(1, int(min_distance))
        coords = peak_local_max(distance, **peak_kwargs)
        markers = np.zeros(binary.shape, dtype=np.int32)
        if coords.size:
            markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1, dtype=np.int32)
        else:
            markers = measure.label(binary, connectivity=1).astype(np.int32)
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labels = measure.label(binary, connectivity=1)

    labels = remove_small_labeled_objects(labels, min_size)
    labels, _fw, _inv = segmentation.relabel_sequential(labels)
    return np.asarray(labels, dtype=np.int32), threshold


def labels_from_binary(
    binary: np.ndarray,
    *,
    min_size: int,
    fill_holes: bool,
    split_touching: bool,
    min_distance: int,
    min_distance_um: float | None = None,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage import measure, segmentation
    from skimage.feature import peak_local_max

    cleaned = remove_small_binary_objects(binary.astype(bool), min_size)
    if fill_holes:
        cleaned = remove_small_binary_holes(cleaned, min_size)
    if not np.any(cleaned):
        return np.zeros(cleaned.shape, dtype=np.int32)

    if split_touching:
        distance = ndi.distance_transform_edt(cleaned, sampling=spacing)
        footprint = _physical_peak_footprint(spacing, min_distance_um, cleaned.ndim)
        peak_kwargs = {
            "labels": cleaned,
            "exclude_border": False,
        }
        if footprint is not None:
            peak_kwargs["footprint"] = footprint
            peak_kwargs["min_distance"] = 1
        else:
            peak_kwargs["min_distance"] = max(1, int(min_distance))
        coords = peak_local_max(distance, **peak_kwargs)
        markers = np.zeros(cleaned.shape, dtype=np.int32)
        if coords.size:
            markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1, dtype=np.int32)
        else:
            markers = measure.label(cleaned, connectivity=1).astype(np.int32)
        labels = segmentation.watershed(-distance, markers, mask=cleaned)
    else:
        labels = measure.label(cleaned, connectivity=1)

    labels = remove_small_labeled_objects(labels, min_size)
    labels, _fw, _inv = segmentation.relabel_sequential(labels)
    return np.asarray(labels, dtype=np.int32)


def _physical_peak_footprint(
    spacing: tuple[float, ...] | None,
    min_distance_um: float | None,
    ndim: int,
) -> np.ndarray | None:
    if spacing is None or min_distance_um is None:
        return None
    radius_um = float(min_distance_um)
    if radius_um <= 0 or not np.isfinite(radius_um):
        return None
    radii = [max(1, int(np.ceil(radius_um / s))) for s in spacing[:ndim]]
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    dist2 = np.zeros(tuple(2 * r + 1 for r in radii), dtype=np.float32)
    for grid, step in zip(grids, spacing[:ndim]):
        dist2 += (grid.astype(np.float32) * float(step)) ** 2
    footprint = dist2 <= radius_um**2
    center = tuple(radii)
    footprint[center] = True
    return footprint


def _xy_filter_size(ndim: int, radius: int) -> tuple[int, ...]:
    width = max(1, int(radius) * 2 + 1)
    if ndim == 3:
        return (1, width, width)
    if ndim == 2:
        return (width, width)
    return tuple(width for _ in range(ndim))
