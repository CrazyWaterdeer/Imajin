from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.paths import normalize_user_path
from imajin.results import record_result, unique_result_path
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
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


def _layer_axes_for_seg(layer: Any, ndim: int) -> str:
    md = getattr(layer, "metadata", None) or {}
    if isinstance(md, dict):
        axes = md.get("axes")
        if isinstance(axes, str):
            stripped = axes.replace("C", "")
            if len(stripped) == ndim:
                return stripped
    if ndim == 4:
        return "TZYX"
    if ndim == 3:
        return "ZYX"
    if ndim == 2:
        return "YX"
    return "".join(f"A{i}" for i in range(ndim))


def _label_qc(masks: np.ndarray) -> dict[str, Any]:
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
    flat = masks.ravel()
    counts = np.bincount(flat, minlength=n + 1)[1:]
    qc["object_area_min"] = int(counts.min())
    qc["object_area_median"] = float(np.median(counts))
    qc["object_area_max"] = int(counts.max())
    return qc


def _label_qc_warnings(masks: np.ndarray) -> list[str]:
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


def _slug(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return text or "segmentation"


def _default_qc_png_path(labels_layer: str) -> Path:
    return unique_result_path("segmentation_qc", f"{_slug(labels_layer)}.png")


def _normalize_uint8(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _project_for_qc(data: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim == 2:
        image_plane = data
    else:
        image_plane = np.max(data, axis=0)
    if masks.ndim == 2:
        mask_plane = masks
    else:
        mask_plane = np.max(masks, axis=0)
    return image_plane, mask_plane


def _voxel_spacing(scale: tuple[float, ...] | None, ndim: int) -> tuple[float, ...] | None:
    if not scale:
        return None
    spacing = tuple(float(s) for s in scale[:ndim])
    if len(spacing) != ndim:
        return None
    if any(s <= 0 or not np.isfinite(s) for s in spacing):
        return None
    return spacing


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


def _min_size_from_physical(
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


def _small_default_qc_skip_reason(image: np.ndarray, masks: np.ndarray) -> str | None:
    image_plane, _mask_plane = _project_for_qc(image, masks)
    if image_plane.ndim < 2:
        return f"QC PNG skipped for non-image plane shape {tuple(image_plane.shape)}."
    height, width = (int(image_plane.shape[-2]), int(image_plane.shape[-1]))
    if min(height, width) < 256:
        return (
            f"QC PNG skipped for small image plane {height}x{width}. "
            "Pass qc_png_path to force saving a tiny diagnostic image."
        )
    return None


def _write_segmentation_qc_png(
    image: np.ndarray,
    masks: np.ndarray,
    path: Path,
) -> None:
    from PIL import Image
    from skimage.segmentation import find_boundaries

    image_plane, mask_plane = _project_for_qc(image, masks)
    base = _normalize_uint8(image_plane)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    labels = np.asarray(mask_plane, dtype=np.int64)
    if labels.size and int(labels.max()) > 0:
        rng = np.random.default_rng(12345)
        colors = rng.integers(
            32,
            256,
            size=(int(labels.max()) + 1, 3),
            dtype=np.uint8,
        ).astype(np.float32)
        colors[0] = 0
        mask = labels > 0
        alpha = 0.38
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colors[labels[mask]]
    boundaries = find_boundaries(mask_plane, mode="outer")
    rgb[boundaries] = np.asarray([255, 64, 0], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def _save_qc_png(
    image: np.ndarray,
    masks: np.ndarray,
    path: Path,
    *,
    labels_layer: str,
    source_layer: str,
    method: str,
    force: bool = False,
) -> tuple[str | None, str | None]:
    if not force:
        reason = _small_default_qc_skip_reason(image, masks)
        if reason:
            return None, reason
    _write_segmentation_qc_png(image, masks, path)
    try:
        record_result(
            "segmentation_qc_png",
            path,
            {
                "labels_layer": labels_layer,
                "source_layer": source_layer,
                "method": method,
            },
        )
    except Exception:
        pass
    return str(path), None


def _threshold_value(
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


def _remove_small_binary_objects(binary: np.ndarray, min_size: int) -> np.ndarray:
    from skimage import morphology
    import inspect

    threshold = max(1, int(min_size))
    params = inspect.signature(morphology.remove_small_objects).parameters
    if "max_size" in params:
        return morphology.remove_small_objects(binary, max_size=max(0, threshold - 1))
    return morphology.remove_small_objects(binary, min_size=threshold)


def _remove_small_binary_holes(binary: np.ndarray, min_size: int) -> np.ndarray:
    from skimage import morphology
    import inspect

    threshold = max(1, int(min_size))
    params = inspect.signature(morphology.remove_small_holes).parameters
    if "max_size" in params:
        return morphology.remove_small_holes(binary, max_size=max(0, threshold - 1))
    return morphology.remove_small_holes(binary, area_threshold=threshold)


def _remove_small_labeled_objects(labels: np.ndarray, min_size: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return arr
    counts = np.bincount(arr.ravel(), minlength=n + 1)
    keep = counts >= max(1, int(min_size))
    keep[0] = False
    return np.where(keep[arr], arr, 0).astype(np.int32)


def _xy_filter_size(ndim: int, radius: int) -> tuple[int, ...]:
    width = max(1, int(radius) * 2 + 1)
    if ndim == 3:
        return (1, width, width)
    if ndim == 2:
        return (width, width)
    return tuple(width for _ in range(ndim))


def _estimate_local_background(
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


def _robust_background_sigma(corrected: np.ndarray) -> float:
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


def _target_object_threshold(
    corrected: np.ndarray,
    *,
    method: str,
    percentile: float,
    min_snr: float,
    noise_sigma: float,
) -> float:
    from skimage import filters

    finite = np.asarray(corrected[np.isfinite(corrected)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("cannot threshold an image with no finite pixels")
    if float(finite.max()) <= float(finite.min()):
        raise ValueError("cannot threshold a constant image")

    key = method.lower().strip().replace("-", "_")
    if key in {"auto", "background_corrected", "target"}:
        try:
            threshold = float(filters.threshold_otsu(finite))
        except ValueError:
            positives = finite[finite > 0]
            threshold = float(np.percentile(positives, 25)) if positives.size else 0.0
    elif key == "percentile":
        threshold = float(np.percentile(finite, percentile))
    elif key == "otsu":
        threshold = float(filters.threshold_otsu(finite))
    elif key == "yen":
        threshold = float(filters.threshold_yen(finite))
    elif key == "li":
        threshold = float(filters.threshold_li(finite))
    elif key == "triangle":
        threshold = float(filters.threshold_triangle(finite))
    else:
        raise ValueError(
            "threshold_method must be one of: auto, percentile, otsu, yen, li, triangle"
        )

    snr_floor = float(min_snr) * float(noise_sigma)
    if np.isfinite(snr_floor) and snr_floor > 0:
        threshold = max(threshold, snr_floor)
    return float(threshold)


def _target_object_qc(
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
    warnings = _label_qc_warnings(labels)

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


def _segment_connected_regions(
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

    threshold = _threshold_value(data, threshold_method, percentile)
    binary = np.isfinite(data) & (data > threshold)
    binary = _remove_small_binary_objects(binary, min_size)
    if fill_holes:
        binary = _remove_small_binary_holes(binary, min_size)

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

    labels = _remove_small_labeled_objects(labels, min_size)
    labels, _fw, _inv = segmentation.relabel_sequential(labels)
    return np.asarray(labels, dtype=np.int32), threshold


def _labels_from_binary(
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

    cleaned = _remove_small_binary_objects(binary.astype(bool), min_size)
    if fill_holes:
        cleaned = _remove_small_binary_holes(cleaned, min_size)
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

    labels = _remove_small_labeled_objects(labels, min_size)
    labels, _fw, _inv = segmentation.relabel_sequential(labels)
    return np.asarray(labels, dtype=np.int32)


@tool(
    description="Segment cells with Cellpose-SAM (generalist pretrained model). "
    "Works on 2D images (YX) and 3D z-stacks (ZYX). 4D (TZYX) and time-series (TYX) "
    "inputs must be reduced to a single timepoint first via extract_timepoint or a "
    "per-frame workflow. Set do_3D=True for true 3D segmentation on Z-stacks. "
    "Use diameter=None for auto-estimation; otherwise specify approximate cell "
    "diameter in pixels.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def cellpose_sam(
    image_layer: str,
    do_3D: bool = False,
    diameter: float | None = None,
    model: str = "cpsam",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"cellpose_sam refuses to run on a time-series layer ({axes}, "
            f"shape {data.shape}). Use extract_timepoint to pick a frame first, "
            "or run a per-frame workflow."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"cellpose_sam expects a 2D (YX) or 3D (ZYX) layer, got shape "
            f"{data.shape}. Reduce to YX/ZYX before calling."
        )

    is_3d_input = data.ndim == 3 and "Z" in axes
    use_3d = bool(do_3D) and is_3d_input
    if do_3D and not is_3d_input:
        # Caller asked for 3D but data is 2D — fall back silently to 2D rather
        # than confusing Cellpose.
        use_3d = False

    cp = _get_cellpose_model(model)
    scale = tuple(float(s) for s in getattr(L, "scale", ()) or ())
    anisotropy = None
    if use_3d and len(scale) >= 3 and scale[1] > 0:
        anisotropy = float(scale[0] / scale[1])
    masks, _flows, _styles = cp.eval(
        data,
        diameter=diameter,
        do_3D=use_3d,
        z_axis=0 if use_3d else None,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
    )
    masks = np.asarray(masks).astype(np.int32)
    qc = _label_qc(masks)
    qc_warnings = _label_qc_warnings(masks)

    out_name = f"{L.name}_masks"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "segmentation_method": "cellpose_sam",
            "model": model,
            "diameter": diameter,
            "do_3D": use_3d,
            "min_size": min_size,
            "max_size_fraction": max_size_fraction,
            "anisotropy": anisotropy,
            "axes": "ZYX" if use_3d else "YX",
            "qc_warnings": qc_warnings,
            **qc,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                data,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="cellpose_sam",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "model": model,
        "diameter": diameter,
        "do_3D": use_3d,
        "min_size": min_size,
        "max_size_fraction": max_size_fraction,
        "anisotropy": anisotropy,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Segment bright reporter-positive cells or regions by intensity "
    "thresholding. This is often more appropriate than Cellpose-SAM for CaLexA/GCaMP "
    "or other reporter images where true cell boundaries are not visible. The output "
    "is Labels matching the input layer shape. Use split_touching=True to watershed "
    "touching blobs into candidate cells; leave it False when merged clusters should "
    "be measured as region-level ROIs.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def segment_intensity_regions(
    image_layer: str,
    threshold_method: str = "otsu",
    percentile: float = 99.0,
    min_size: int = 128,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_intensity_regions refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). Use extract_timepoint or a per-frame "
            "workflow first."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"segment_intensity_regions expects a 2D (YX) or 3D (ZYX) layer, got "
            f"shape {data.shape}."
        )
    spacing = _voxel_spacing(tuple(L.scale), data.ndim)
    effective_min_size = _min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=data.ndim,
    ) or int(min_size)

    masks, threshold = _segment_connected_regions(
        data,
        threshold_method=threshold_method,
        percentile=percentile,
        min_size=effective_min_size,
        smoothing_sigma=smoothing_sigma,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )
    qc = _label_qc(masks)
    qc_warnings = _label_qc_warnings(masks)

    out_name = f"{L.name}_regions"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "segmentation_method": "intensity_regions",
            "threshold_method": threshold_method,
            "threshold": threshold,
            "percentile": percentile,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "min_area_um2": min_area_um2,
            "min_volume_um3": min_volume_um3,
            "smoothing_sigma": smoothing_sigma,
            "fill_holes": fill_holes,
            "split_touching": split_touching,
            "min_distance": min_distance,
            "min_distance_um": min_distance_um,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            **qc,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                data,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="intensity_regions",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "n_regions": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "threshold_method": threshold_method,
        "threshold": threshold,
        "percentile": percentile,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Default target-channel segmentation for routine intensity analysis. "
    "Segments target-positive objects/ROIs from a 2D image or 3D z-stack using local "
    "background correction, background-corrected signal thresholding, size cleanup, "
    "and optional watershed splitting. This tool does not try to decide whether the "
    "objects are cells, nuclei, membranes, or clusters; it returns neutral measured "
    "objects suitable for intensity/area/count analysis.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def segment_target_objects(
    image_layer: str,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    from skimage import filters

    L = call_on_main(snapshot_layer, image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)

    axes = _layer_axes_for_seg(L, data.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_target_objects refuses to run on a time-series layer "
            f"({axes}, shape {data.shape}). Use extract_timepoint or a per-frame "
            "workflow first."
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"segment_target_objects expects a 2D (YX) or 3D (ZYX) layer, got "
            f"shape {data.shape}."
        )

    raw = np.asarray(data, dtype=np.float32)
    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    xy_area = int(np.prod(raw.shape[-2:])) if raw.ndim >= 2 else int(raw.size)
    physical_min_size = _min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=raw.ndim,
    )
    effective_min_size = physical_min_size or max(16, min(512, int(round(xy_area * 0.00005))))
    background = _estimate_local_background(
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
        corrected_for_threshold = filters.gaussian(
            corrected,
            sigma=sigma,
            preserve_range=True,
        ).astype(np.float32)
    else:
        corrected_for_threshold = corrected

    noise_sigma = _robust_background_sigma(corrected_for_threshold)
    threshold = _target_object_threshold(
        corrected_for_threshold,
        method=threshold_method,
        percentile=threshold_percentile,
        min_snr=min_snr,
        noise_sigma=noise_sigma,
    )

    high_threshold = max(float(threshold), float(high_snr) * float(noise_sigma))
    if high_threshold > threshold and np.any(corrected_for_threshold >= high_threshold):
        binary = filters.apply_hysteresis_threshold(
            corrected_for_threshold,
            low=float(threshold),
            high=float(high_threshold),
        )
    else:
        binary = corrected_for_threshold > float(threshold)

    masks = _labels_from_binary(
        binary,
        min_size=effective_min_size,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )
    qc = _label_qc(masks)
    signal_qc, qc_warnings = _target_object_qc(
        raw,
        corrected_for_threshold,
        masks,
        noise_sigma=noise_sigma,
    )

    out_name = f"{L.name}_objects"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "segmentation_method": "target_objects",
            "object_unit": "object_or_roi",
            "background_radius": background_radius,
            "background_method": background_method,
            "background_percentile": background_percentile,
            "threshold_method": threshold_method,
            "threshold": threshold,
            "high_threshold": high_threshold,
            "threshold_percentile": threshold_percentile,
            "min_snr": min_snr,
            "high_snr": high_snr,
            "noise_sigma": noise_sigma,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "min_area_um2": min_area_um2,
            "min_volume_um3": min_volume_um3,
            "smoothing_sigma": smoothing_sigma,
            "fill_holes": fill_holes,
            "split_touching": split_touching,
            "min_distance": min_distance,
            "min_distance_um": min_distance_um,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            **qc,
            **signal_qc,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="target_objects",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "object_unit": "object_or_roi",
        "n_objects": qc["n_objects"],
        "n_cells": qc["n_objects"],
        "shape": qc["shape"],
        "dtype": qc["dtype"],
        "background_radius": background_radius,
        "background_method": background_method,
        "background_percentile": background_percentile,
        "threshold_method": threshold_method,
        "threshold": threshold,
        "high_threshold": high_threshold,
        "threshold_percentile": threshold_percentile,
        "min_snr": min_snr,
        "high_snr": high_snr,
        "noise_sigma": noise_sigma,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
        "voxel_spacing": list(spacing) if spacing is not None else None,
        "axes": axes,
        "empty_mask": qc["empty_mask"],
        "object_area_min": qc["object_area_min"],
        "object_area_median": qc["object_area_median"],
        "object_area_max": qc["object_area_max"],
        **signal_qc,
        "qc_warnings": qc_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }
