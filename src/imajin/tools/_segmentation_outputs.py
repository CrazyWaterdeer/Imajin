from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from imajin.result_bundles import bundle_output_path, register_output


def _slug(value: str) -> str:
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
    return bundle_output_path("qc", f"{_slug(labels_layer)}.png")


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
    _write_segmentation_qc_png(
        image,
        masks,
        path,
        secondary_outline_mask=secondary_outline_mask,
    )
    try:
        register_output(
            "qc_png",
            path,
            {
                "labels_layer": labels_layer,
                "source_layer": source_layer,
                "method": method,
            },
        )
    except ValueError:
        pass
    return str(path), None
