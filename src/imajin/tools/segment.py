from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.analysis.segmentation import (
    dilate_binary_um as _dilate_binary_um,
    estimate_local_background as _estimate_local_background,
    intersect_labels_with_mask as _intersect_labels_with_mask,
    label_qc as _label_qc,
    label_qc_warnings as _label_qc_warnings,
    labels_from_binary as _labels_from_binary,
    min_size_from_physical as _min_size_from_physical,
    remove_small_binary_objects as _remove_small_binary_objects,
    robust_background_sigma as _robust_background_sigma,
    segment_connected_regions as _segment_connected_regions,
    target_object_qc as _target_object_qc,
    target_object_threshold as _target_object_threshold,
    threshold_noise_floor as _threshold_noise_floor,
    voxel_spacing as _voxel_spacing,
)
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
    return layer_axes_from_metadata(md, ndim, default_3d="ZYX")


def _slug(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return text or "segmentation"


def _unique_file(root: Path, filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    i = 2
    while True:
        candidate = root / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _source_path_from_layer(layer: Any) -> str | None:
    md = getattr(layer, "metadata", None)
    if not isinstance(md, dict):
        return None
    raw = md.get("source_path") or md.get("path")
    return str(raw) if raw else None


def _source_metadata_from_layer(layer: Any) -> dict[str, str]:
    source = _source_path_from_layer(layer)
    if not source:
        return {}
    return {"source_path": source, "path": source}


def _default_qc_png_path(labels_layer: str, source_layer: Any | None = None) -> Path:
    if source_layer is not None:
        source = _source_path_from_layer(source_layer)
        if source:
            from imajin.anchor import resolve_anchor_folder

            anchor = resolve_anchor_folder([source])
            if anchor is not None:
                return _unique_file(
                    anchor / "segmentation_qc",
                    f"{_slug(labels_layer)}.png",
                )
    return unique_result_path("segmentation_qc", f"{_slug(labels_layer)}.png")


def _saturation_warnings(data: Any, *, layer_name: str) -> list[str]:
    arr = np.asarray(data)
    if arr.size == 0:
        return []
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return []

    finite = arr[finite_mask]
    total = int(finite.size)
    warnings: list[str] = []
    if np.issubdtype(arr.dtype, np.integer):
        dtype_max = np.iinfo(arr.dtype).max
        saturated = int(np.count_nonzero(finite >= dtype_max))
        threshold = max(16, int(np.ceil(total * 0.001)))
        if saturated >= threshold:
            warnings.append(
                f"{layer_name}: {saturated / total:.2%} of finite pixels are at "
                f"the dtype maximum ({dtype_max}); intensity segregation may be "
                "limited by saturation"
            )
        return warnings

    values = np.asarray(finite, dtype=np.float32)
    vmax = float(np.max(values))
    if not np.isfinite(vmax) or vmax <= 0:
        return []
    saturated = int(np.count_nonzero(values >= vmax))
    threshold = max(64, int(np.ceil(total * 0.01)))
    if saturated >= threshold:
        warnings.append(
            f"{layer_name}: {saturated / total:.2%} of finite pixels share the "
            "observed maximum; check for clipping/saturation before interpreting "
            "intensity tiers"
        )
    return warnings


def _domain_smoothing_sigma(
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


def _smooth_domain_image(
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
    sigma = _domain_smoothing_sigma(spacing, raw.ndim, smooth_sigma_um)
    return ndi.gaussian_filter(work, sigma=sigma).astype(np.float32, copy=False)


def _domain_min_size_from_physical(
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
    return _min_size_from_physical(
        min_size=None,
        min_volume_um3=volume,
        min_area_um2=area,
        spacing=spacing,
        ndim=ndim,
    )


def _domain_physical_sizes(
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


def _filter_domain_components(
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
    *,
    secondary_outline_mask: np.ndarray | None = None,
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

    if secondary_outline_mask is not None:
        _, secondary_plane = _project_for_qc(image, secondary_outline_mask)
        secondary_boundaries = find_boundaries(secondary_plane, mode="outer")
        rgb[secondary_boundaries] = np.asarray([0, 200, 220], dtype=np.uint8)

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
    secondary_outline_mask: np.ndarray | None = None,
) -> tuple[str | None, str | None]:
    if not force:
        reason = _small_default_qc_skip_reason(image, masks)
        if reason:
            return None, reason
    _write_segmentation_qc_png(image, masks, path, secondary_outline_mask=secondary_outline_mask)
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


def _target_threshold_for_scope(
    corrected: np.ndarray,
    *,
    threshold_method: str,
    threshold_percentile: float,
    min_snr: float,
    boundary_mask: np.ndarray | None = None,
) -> tuple[float, float, str, list[str]]:
    full_noise_sigma = _robust_background_sigma(corrected)
    warnings: list[str] = []
    if boundary_mask is None:
        return (
            _target_object_threshold(
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
            _target_object_threshold(
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
            _target_object_threshold(
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

    scoped_noise_sigma = _robust_background_sigma(scoped_values)
    return (
        _target_object_threshold(
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
    data = materialize_array(L.data)

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
            **_source_metadata_from_layer(L),
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
                else _default_qc_png_path(layer.name, L)
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
    data = materialize_array(L.data)

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
            **_source_metadata_from_layer(L),
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
                else _default_qc_png_path(layer.name, L)
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
    boundary_mask: str | None = None,
) -> dict[str, Any]:
    from skimage import filters

    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)

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

    # Load the boundary mask early so it can be used as the hysteresis grow region.
    boundary_data_bool: np.ndarray | None = None
    if boundary_mask is not None:
        boundary_layer_snapshot = call_on_main(snapshot_layer, boundary_mask)
        _boundary_raw = materialize_array(boundary_layer_snapshot.data)
        if _boundary_raw.shape != raw.shape:
            raise ValueError(
                f"boundary_mask shape {_boundary_raw.shape} does not match "
                f"target image shape {raw.shape}"
            )
        boundary_data_bool = _boundary_raw > 0

    threshold, noise_sigma, threshold_scope, threshold_warnings = _target_threshold_for_scope(
        corrected_for_threshold,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        min_snr=min_snr,
        boundary_mask=boundary_data_bool,
    )

    high_threshold = max(float(threshold), float(high_snr) * float(noise_sigma))
    if boundary_data_bool is not None:
        scoped_threshold_image = np.where(
            boundary_data_bool,
            corrected_for_threshold,
            -np.inf,
        ).astype(np.float32, copy=False)
        low_candidates = (scoped_threshold_image >= float(threshold)) & boundary_data_bool
        high_seeds = (scoped_threshold_image >= high_threshold) & boundary_data_bool
        if high_threshold > threshold and np.any(high_seeds):
            binary = (
                filters.apply_hysteresis_threshold(
                    scoped_threshold_image,
                    low=float(threshold),
                    high=float(high_threshold),
                )
                & boundary_data_bool
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
    qc_warnings = saturation_warnings + threshold_warnings + qc_warnings

    if boundary_data_bool is not None:
        masks = _intersect_labels_with_mask(
            masks, boundary_data_bool, renumber=True
        )
        qc = _label_qc(masks)
        signal_qc, qc_warnings = _target_object_qc(
            raw,
            corrected_for_threshold,
            masks,
            noise_sigma=noise_sigma,
        )
        qc_warnings = saturation_warnings + threshold_warnings + qc_warnings

    out_name = f"{L.name}_objects"
    layer = call_on_main(
        add_labels_from_worker,
        masks,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
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
            "threshold_scope": threshold_scope,
            "min_size": effective_min_size,
            "requested_min_size": min_size,
            "min_area_um2": min_area_um2,
            "min_volume_um3": min_volume_um3,
            "smoothing_sigma": smoothing_sigma,
            "fill_holes": fill_holes,
            "split_touching": split_touching,
            "min_distance": min_distance,
            "min_distance_um": min_distance_um,
            "boundary_mask": boundary_mask,
            "voxel_spacing": spacing,
            "axes": "ZYX" if data.ndim == 3 else "YX",
            "qc_warnings": qc_warnings,
            **qc,
            **signal_qc,
        },
    )

    secondary_mask_array: np.ndarray | None = None
    if boundary_mask is not None:
        bm_snapshot = call_on_main(snapshot_layer, boundary_mask)
        bm_data = materialize_array(bm_snapshot.data)
        secondary_mask_array = (bm_data > 0).astype(np.int32)

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="target_objects",
                force=qc_png_path is not None,
                secondary_outline_mask=secondary_mask_array,
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
        "threshold_scope": threshold_scope,
        "min_size": effective_min_size,
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": smoothing_sigma,
        "fill_holes": fill_holes,
        "split_touching": split_touching,
        "min_distance": min_distance,
        "min_distance_um": min_distance_um,
        "boundary_mask": boundary_mask,
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


@tool(
    description="Segment a permissive expression domain on a reporter channel using "
    "a noise-floor threshold (median + k*MAD of the dark percentile). No background "
    "subtraction, so cluster interiors are preserved. Use as Tier 1 of two-tier "
    "expression analyses where baseline expression must be captured alongside "
    "active sub-objects. By default, connected components are cleaned and merged "
    "into one binary domain label for compact domain-level intensity measurement.",
    phase="2",
    worker=True,
)
def segment_expression_domain(
    image_layer: str,
    threshold_strategy: str = "noise_floor",
    k_mad: float = 5.25,
    dark_percentile: float = 10.0,
    counterstain_layer: str | None = None,
    counterstain_dilation_um: float = 0.0,
    is_nuclear: bool | None = None,
    min_area_um2: float = 5.0,
    min_volume_um3: float | None = None,
    smooth_sigma_um: float = 0.5,
    max_components: int = 256,
    min_component_fraction: float = 0.0,
    merge_components: bool = True,
    dilation_um: float = 0.0,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    if threshold_strategy != "noise_floor":
        raise ValueError(
            f"threshold_strategy must be 'noise_floor' (got {threshold_strategy!r})"
        )

    L = call_on_main(snapshot_layer, image_layer)
    data = materialize_array(L.data)
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)
    raw = np.asarray(data, dtype=np.float32)

    axes = _layer_axes_for_seg(L, raw.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_expression_domain refuses to run on a time-series layer "
            f"({axes}, shape {raw.shape})."
        )
    if raw.ndim < 2 or raw.ndim > 3:
        raise ValueError(
            f"segment_expression_domain expects 2D (YX) or 3D (ZYX), got {raw.shape}."
        )

    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    threshold_image = _smooth_domain_image(
        raw,
        spacing=spacing,
        smooth_sigma_um=smooth_sigma_um,
    )
    threshold = _threshold_noise_floor(
        threshold_image, k_mad=k_mad, dark_percentile=dark_percentile
    )
    binary = np.isfinite(raw) & np.isfinite(threshold_image) & (
        threshold_image > threshold
    )

    counterstain_used = False
    counterstain_warnings: list[str] = []
    if counterstain_layer is not None:
        if not is_nuclear:
            counterstain_warnings.append(
                "counterstain marker is non-nuclear or unknown; reporter-only "
                "domain used"
            )
        else:
            cs_layer = call_on_main(snapshot_layer, counterstain_layer)
            cs_data = materialize_array(cs_layer.data, dtype=np.float32)
            if cs_data.shape != raw.shape:
                counterstain_warnings.append(
                    f"counterstain shape {cs_data.shape} differs from reporter "
                    f"shape {raw.shape}; counterstain ignored"
                )
            else:
                from skimage import filters as _filters
                cs_finite = cs_data[np.isfinite(cs_data)]
                if cs_finite.size and float(cs_finite.max()) > float(cs_finite.min()):
                    cs_threshold = float(_filters.threshold_otsu(cs_finite))
                    cs_binary = np.isfinite(cs_data) & (cs_data > cs_threshold)
                    if counterstain_dilation_um > 0 and spacing is not None:
                        cs_binary = _dilate_binary_um(
                            cs_binary,
                            spacing=spacing,
                            radius_um=counterstain_dilation_um,
                        )
                    binary = binary & cs_binary
                    counterstain_used = True
                else:
                    counterstain_warnings.append(
                        "counterstain has no usable signal; reporter-only domain used"
                    )

    physical_min_size = _domain_min_size_from_physical(
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
        ndim=raw.ndim,
    )
    if physical_min_size:
        binary = _remove_small_binary_objects(binary, physical_min_size)

    if dilation_um > 0 and spacing is not None:
        binary = _dilate_binary_um(binary, spacing=spacing, radius_um=dilation_um)

    labels, component_stats, component_warnings = _filter_domain_components(
        binary,
        max_components=max_components,
        min_component_fraction=min_component_fraction,
        merge_components=merge_components,
    )
    domain_warnings = saturation_warnings + component_warnings
    size_stats = _domain_physical_sizes(labels > 0, spacing)
    n_components = int(component_stats["n_components_retained"])
    domain_label_count = int(component_stats["domain_label_count"])

    if domain_label_count == 0:
        empty = np.zeros(raw.shape, dtype=np.int32)
        out_name = f"{L.name}_domain"
        layer = call_on_main(
            add_labels_from_worker,
            empty,
            name=out_name,
            scale=tuple(L.scale),
            metadata={
                "source_layer": L.name,
                **_source_metadata_from_layer(L),
                "segmentation_method": "expression_domain",
                "noise_floor_threshold": float(threshold),
                "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
                "smooth_sigma_um": float(smooth_sigma_um),
                "k_mad": float(k_mad),
                "dark_percentile": float(dark_percentile),
                "counterstain_used": counterstain_used,
                "counterstain_warnings": counterstain_warnings,
                "domain_warnings": domain_warnings,
                "min_area_um2": float(min_area_um2),
                "min_volume_um3": min_volume_um3,
                "min_size_voxels": physical_min_size,
                "max_components": max_components,
                "min_component_fraction": float(min_component_fraction),
                "merge_components": bool(merge_components),
                **component_stats,
                **size_stats,
                "empty_mask": True,
            },
        )
        return {
            "labels_layer": layer.name,
            "n_components": 0,
            "domain_label_count": 0,
            "domain_area_um2": 0.0,
            "domain_volume_um3": 0.0 if raw.ndim == 3 else None,
            "domain_voxels": 0,
            "noise_floor_threshold": float(threshold),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "qc_png_path": None,
            "qc_png_error": None,
            "qc_png_skipped_reason": "empty mask",
            "empty_mask": True,
        }

    domain_area_um2 = float(size_stats["domain_area_um2"])
    domain_volume_um3 = size_stats["domain_volume_um3"]

    out_name = f"{L.name}_domain"
    layer = call_on_main(
        add_labels_from_worker,
        labels,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "expression_domain",
            "noise_floor_threshold": float(threshold),
            "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
            "smooth_sigma_um": float(smooth_sigma_um),
            "k_mad": float(k_mad),
            "dark_percentile": float(dark_percentile),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "n_components": n_components,
            "domain_label_count": domain_label_count,
            "domain_area_um2": domain_area_um2,
            "domain_volume_um3": domain_volume_um3,
            "domain_voxels": int(size_stats["domain_voxels"]),
            "min_area_um2": float(min_area_um2),
            "min_volume_um3": min_volume_um3,
            "min_size_voxels": physical_min_size,
            "max_components": max_components,
            "min_component_fraction": float(min_component_fraction),
            "merge_components": bool(merge_components),
            **component_stats,
            "empty_mask": False,
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
                else _default_qc_png_path(layer.name, L)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                labels,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="expression_domain",
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
        "n_components": n_components,
        "domain_label_count": domain_label_count,
        "domain_area_um2": domain_area_um2,
        "domain_volume_um3": domain_volume_um3,
        "domain_voxels": int(size_stats["domain_voxels"]),
        "noise_floor_threshold": float(threshold),
        "counterstain_used": counterstain_used,
        "counterstain_warnings": counterstain_warnings,
        "domain_warnings": domain_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
        "empty_mask": False,
    }
