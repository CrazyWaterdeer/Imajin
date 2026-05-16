"""ROI correction from user markings on a MIP overlay.

Phase 2 of the SNR/ROI initiative. The user reviews an automatic ROI on a
max-intensity projection (MIP) and marks pixels to add or remove. This
module rebuilds the ROI on the original 3D z-stack from those markings.

Conventions
-----------
* `corrected` is the background-corrected target image (2D YX or 3D ZYX).
* `auto_labels` has the same shape and is an integer label image (0=bg).
* `add_points` / `remove_points` are sequences of (y, x) integer coords on
  the YX plane. For a 3D image we pick z* = argmax_z corrected[z, y, x]
  before applying the per-point operation.
* `add_regions` / `remove_regions` are 2D boolean masks of shape (Y, X).
  For a 3D image they are broadcast across every z slice.

The function returns the new integer label image and a small info dict
useful for QC / UI feedback. No napari imports here so the algorithm can
be tested headlessly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from imajin.analysis.segmentation import labels_from_binary, robust_background_sigma


Point2D = tuple[int, int]


def _is_3d(arr: np.ndarray) -> bool:
    return arr.ndim == 3


def _resolve_seed(corrected: np.ndarray, y: int, x: int) -> tuple[int, ...] | None:
    """Return the ND seed coordinate for a (y, x) point.

    For 3D inputs we choose the z slice with the maximum corrected value at
    (y, x). Returns ``None`` when the column has no finite values.
    """
    H, W = corrected.shape[-2:]
    if not (0 <= y < H and 0 <= x < W):
        return None
    if _is_3d(corrected):
        column = corrected[:, y, x]
        finite = np.isfinite(column)
        if not np.any(finite):
            return None
        masked = np.where(finite, column, -np.inf)
        z_star = int(np.argmax(masked))
        return (z_star, int(y), int(x))
    return (int(y), int(x))


def _broadcast_region(region2d: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a YX boolean mask to match a 2D or 3D target shape."""
    region2d = np.asarray(region2d, dtype=bool)
    if region2d.shape != shape[-2:]:
        raise ValueError(
            f"region mask shape {region2d.shape} does not match image YX "
            f"shape {shape[-2:]}"
        )
    if len(shape) == 3:
        return np.broadcast_to(region2d[None, :, :], shape).copy()
    return region2d


def _component_mask(mask: np.ndarray, seed: tuple[int, ...]) -> np.ndarray:
    """Return the connected component of ``mask`` that contains ``seed``.

    Empty mask if the seed pixel is False.
    """
    from scipy import ndimage as ndi

    if not bool(mask[seed]):
        return np.zeros(mask.shape, dtype=bool)
    labeled, _ = ndi.label(mask)
    target = int(labeled[seed])
    if target == 0:
        return np.zeros(mask.shape, dtype=bool)
    return labeled == target


def _flood_fill_from_seed(
    corrected: np.ndarray,
    seed: tuple[int, ...],
    *,
    floor: float,
) -> np.ndarray:
    """Connected component of ``corrected > floor`` containing ``seed``."""
    above = (corrected > floor) & np.isfinite(corrected)
    if not above[seed]:
        # The user pointed at a pixel below the floor. Lower the floor just
        # enough to include the seed, then take the component.
        seed_val = float(corrected[seed])
        if not np.isfinite(seed_val):
            return np.zeros(corrected.shape, dtype=bool)
        above = (corrected > min(floor, seed_val - 1e-6)) & np.isfinite(corrected)
    return _component_mask(above, seed)


def correct_roi_from_markings(
    auto_labels: np.ndarray,
    corrected: np.ndarray,
    *,
    add_points: Sequence[Point2D] = (),
    remove_points: Sequence[Point2D] = (),
    add_regions: Sequence[np.ndarray] = (),
    remove_regions: Sequence[np.ndarray] = (),
    noise_sigma: float | None = None,
    base_threshold: float = 0.0,
    add_seed_growth_k_snr: float = 1.5,
    region_min_snr_scale: float = 0.5,
    min_size: int = 16,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    spacing: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild ROI labels from automatic labels + user markings.

    Parameters
    ----------
    auto_labels
        Integer label image (2D YX or 3D ZYX), 0 = background.
    corrected
        Background-corrected target intensity image, same shape.
    add_points, remove_points
        Sequences of (y, x) on the YX plane.
    add_regions, remove_regions
        Sequences of 2D boolean masks on the YX plane.
    noise_sigma
        Robust background sigma. If ``None``, computed from ``corrected``.
    base_threshold
        Threshold value used for the original auto segmentation. Used as
        the upper bound when scaling the threshold inside add_regions.
    add_seed_growth_k_snr
        Floor for the add-point flood-fill is ``k * noise_sigma``. Keep
        conservative so a click does not flood the entire image.
    region_min_snr_scale
        Inside an add_region we accept ``corrected > scale * base_threshold``
        (with the same SNR floor) so dim signal that auto-segmentation
        missed can be recovered.
    min_size, fill_holes, split_touching, min_distance, min_distance_um, spacing
        Forwarded to :func:`labels_from_binary` for the final relabeling.
    """
    auto_labels = np.asarray(auto_labels)
    corrected = np.asarray(corrected, dtype=np.float32)
    if auto_labels.shape != corrected.shape:
        raise ValueError(
            f"auto_labels shape {auto_labels.shape} does not match corrected "
            f"shape {corrected.shape}"
        )
    if corrected.ndim not in (2, 3):
        raise ValueError(
            f"correct_roi_from_markings expects 2D or 3D input, got "
            f"{corrected.ndim}D"
        )

    if noise_sigma is None:
        noise_sigma = robust_background_sigma(corrected)
    sigma = float(noise_sigma) if np.isfinite(noise_sigma) else 0.0

    base_mask = auto_labels > 0
    add_mask = np.zeros(corrected.shape, dtype=bool)
    remove_mask = np.zeros(corrected.shape, dtype=bool)

    info: dict[str, Any] = {
        "noise_sigma": sigma,
        "add_points": len(add_points),
        "remove_points": len(remove_points),
        "add_regions": len(add_regions),
        "remove_regions": len(remove_regions),
        "add_points_voxels": 0,
        "remove_points_voxels": 0,
        "add_regions_voxels": 0,
        "remove_regions_voxels": 0,
        "skipped_points": 0,
    }

    growth_floor = max(float(add_seed_growth_k_snr) * sigma, 0.0)

    for y, x in add_points:
        seed = _resolve_seed(corrected, int(y), int(x))
        if seed is None:
            info["skipped_points"] += 1
            continue
        grown = _flood_fill_from_seed(corrected, seed, floor=growth_floor)
        info["add_points_voxels"] += int(grown.sum())
        add_mask |= grown

    for y, x in remove_points:
        seed = _resolve_seed(corrected, int(y), int(x))
        if seed is None:
            info["skipped_points"] += 1
            continue
        candidate = base_mask | add_mask
        comp = _component_mask(candidate, seed)
        info["remove_points_voxels"] += int(comp.sum())
        remove_mask |= comp

    region_threshold = max(
        float(region_min_snr_scale) * float(base_threshold),
        growth_floor,
    )
    for region2d in add_regions:
        region3d = _broadcast_region(region2d, corrected.shape)
        addition = region3d & (corrected > region_threshold) & np.isfinite(corrected)
        info["add_regions_voxels"] += int(addition.sum())
        add_mask |= addition

    for region2d in remove_regions:
        region3d = _broadcast_region(region2d, corrected.shape)
        info["remove_regions_voxels"] += int(region3d.sum())
        remove_mask |= region3d

    composed = (base_mask | add_mask) & ~remove_mask

    labels = labels_from_binary(
        composed,
        min_size=int(min_size),
        fill_holes=bool(fill_holes),
        split_touching=bool(split_touching),
        min_distance=int(min_distance),
        min_distance_um=min_distance_um,
        spacing=spacing,
    )

    info["final_voxels"] = int((labels > 0).sum())
    info["final_objects"] = int(labels.max())
    return labels, info
