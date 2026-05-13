from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from imajin.analysis.segmentation import (
    estimate_local_background,
    label_qc,
    labels_from_binary,
    min_size_from_physical,
    remove_small_labeled_objects,
    robust_background_sigma,
    target_object_qc,
)
from imajin.analysis.target_segmentation import target_threshold_for_scope


@dataclass(frozen=True)
class SegmentationCandidate:
    name: str
    strategy: str
    labels: np.ndarray
    params: dict[str, Any]
    metrics: dict[str, Any]
    warnings: list[str]
    score: float


def segment_target_array(
    image: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    background_radius: int,
    background_method: str,
    background_percentile: float,
    threshold_method: str,
    threshold_percentile: float,
    min_snr: float,
    high_snr: float,
    min_size: int | None,
    min_area_um2: float | None,
    min_volume_um3: float | None,
    smoothing_sigma: float,
    fill_holes: bool,
    split_touching: bool,
    min_distance: int,
    min_distance_um: float | None,
    boundary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment a 2D/3D target image without creating viewer layers."""

    from skimage import filters

    raw = np.asarray(image, dtype=np.float32)
    physical_min_size = min_size_from_physical(
        min_size=min_size,
        min_volume_um3=min_volume_um3,
        min_area_um2=min_area_um2,
        spacing=spacing,
        ndim=raw.ndim,
    )
    xy_area = int(np.prod(raw.shape[-2:])) if raw.ndim >= 2 else int(raw.size)
    effective_min_size = physical_min_size or max(16, min(512, int(round(xy_area * 0.00005))))

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
        corrected_for_threshold = filters.gaussian(
            corrected,
            sigma=sigma,
            preserve_range=True,
        ).astype(np.float32)
    else:
        corrected_for_threshold = corrected

    boundary_bool = None if boundary_mask is None else np.asarray(boundary_mask, dtype=bool)
    threshold, noise_sigma, threshold_scope, threshold_warnings = target_threshold_for_scope(
        corrected_for_threshold,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        min_snr=min_snr,
        boundary_mask=boundary_bool,
    )
    high_threshold = max(float(threshold), float(high_snr) * float(noise_sigma))

    if boundary_bool is not None:
        scoped_threshold_image = np.where(
            boundary_bool,
            corrected_for_threshold,
            -np.inf,
        ).astype(np.float32, copy=False)
        low_candidates = (scoped_threshold_image >= float(threshold)) & boundary_bool
        high_seeds = (scoped_threshold_image >= high_threshold) & boundary_bool
        if high_threshold > threshold and np.any(high_seeds):
            binary = (
                filters.apply_hysteresis_threshold(
                    scoped_threshold_image,
                    low=float(threshold),
                    high=float(high_threshold),
                )
                & boundary_bool
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
        min_size=effective_min_size,
        fill_holes=fill_holes,
        split_touching=split_touching,
        min_distance=min_distance,
        min_distance_um=min_distance_um,
        spacing=spacing,
    )

    metadata = {
        "background_radius": int(background_radius),
        "background_method": background_method,
        "background_percentile": float(background_percentile),
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "high_threshold": float(high_threshold),
        "threshold_percentile": float(threshold_percentile),
        "min_snr": float(min_snr),
        "high_snr": float(high_snr),
        "noise_sigma": float(noise_sigma),
        "threshold_scope": threshold_scope,
        "min_size": int(effective_min_size),
        "requested_min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": float(smoothing_sigma),
        "fill_holes": bool(fill_holes),
        "split_touching": bool(split_touching),
        "min_distance": int(min_distance),
        "min_distance_um": min_distance_um,
        "threshold_warnings": list(threshold_warnings),
        "corrected_for_threshold": corrected_for_threshold,
    }
    return np.asarray(masks, dtype=np.int32), metadata


def segment_target_array_plane_stitch(
    image: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    background_radius: int,
    background_method: str,
    background_percentile: float,
    threshold_method: str,
    threshold_percentile: float,
    min_snr: float,
    high_snr: float,
    min_size: int | None,
    min_area_um2: float | None,
    min_volume_um3: float | None,
    smoothing_sigma: float,
    fill_holes: bool,
    split_touching: bool,
    min_distance: int,
    min_distance_um: float | None,
    stitch_min_overlap: float,
    stitch_max_centroid_distance: float | None,
    stitch_max_area_ratio: float,
    boundary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment each Z plane independently, then stitch 2D ROIs into 3D labels."""

    raw = np.asarray(image, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"plane-stitch segmentation expects ZYX data, got {raw.shape}")

    plane_spacing = spacing[-2:] if spacing is not None and len(spacing) >= 2 else None
    plane_labels: list[np.ndarray] = []
    plane_records: list[dict[str, Any]] = []
    for z in range(raw.shape[0]):
        plane_boundary = None if boundary_mask is None else np.asarray(boundary_mask[z], dtype=bool)
        labels_2d, record = segment_target_array(
            raw[z],
            spacing=plane_spacing,
            background_radius=background_radius,
            background_method=background_method,
            background_percentile=background_percentile,
            threshold_method=threshold_method,
            threshold_percentile=threshold_percentile,
            min_snr=min_snr,
            high_snr=high_snr,
            min_size=min_size,
            min_area_um2=min_area_um2,
            min_volume_um3=None,
            smoothing_sigma=smoothing_sigma,
            fill_holes=fill_holes,
            split_touching=split_touching,
            min_distance=min_distance,
            min_distance_um=min_distance_um,
            boundary_mask=plane_boundary,
        )
        plane_labels.append(labels_2d)
        plane_records.append(
            {
                "z": int(z),
                "n_objects": int(labels_2d.max()) if labels_2d.size else 0,
                "threshold": record.get("threshold"),
                "noise_sigma": record.get("noise_sigma"),
                "threshold_scope": record.get("threshold_scope"),
                "threshold_warnings": record.get("threshold_warnings", []),
            }
        )

    labels_stack = np.stack(plane_labels, axis=0).astype(np.int32)
    stitched, stitch_record = stitch_plane_labels(
        labels_stack,
        min_overlap_fraction=stitch_min_overlap,
        max_centroid_distance=stitch_max_centroid_distance,
        max_area_ratio=stitch_max_area_ratio,
    )

    if min_volume_um3 is not None and spacing is not None:
        min_volume_voxels = min_size_from_physical(
            min_size=None,
            min_volume_um3=min_volume_um3,
            min_area_um2=None,
            spacing=spacing,
            ndim=3,
        )
        if min_volume_voxels is not None:
            stitched = remove_small_labeled_objects(stitched, min_volume_voxels)
            stitched = _relabel_sequential(stitched)

    metadata = {
        "plane_records": plane_records,
        "plane_label_count": int(labels_stack.max()) if labels_stack.size else 0,
        "stitch": stitch_record,
        "threshold_method": threshold_method,
        "threshold_percentile": float(threshold_percentile),
        "min_snr": float(min_snr),
        "high_snr": float(high_snr),
        "min_size": min_size,
        "min_area_um2": min_area_um2,
        "min_volume_um3": min_volume_um3,
        "smoothing_sigma": float(smoothing_sigma),
        "split_touching": bool(split_touching),
    }
    return np.asarray(stitched, dtype=np.int32), metadata


def stitch_plane_labels(
    labels_stack: np.ndarray,
    *,
    min_overlap_fraction: float = 0.2,
    max_centroid_distance: float | None = None,
    max_area_ratio: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(labels_stack, dtype=np.int32)
    if arr.ndim != 3:
        raise ValueError(f"stitch_plane_labels expects ZYX labels, got {arr.shape}")

    nodes: dict[tuple[int, int], int] = {}
    areas: dict[tuple[int, int], int] = {}
    centroids: dict[tuple[int, int], tuple[float, float]] = {}
    next_id = 0
    for z in range(arr.shape[0]):
        plane = arr[z]
        labels = np.unique(plane)
        labels = labels[labels > 0]
        for label in labels:
            key = (int(z), int(label))
            nodes[key] = next_id
            next_id += 1
            yy, xx = np.nonzero(plane == int(label))
            areas[key] = int(yy.size)
            centroids[key] = (float(np.mean(yy)), float(np.mean(xx)))

    parent = list(range(next_id))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    edges: list[dict[str, Any]] = []
    ambiguous_pairs = 0
    for z in range(arr.shape[0] - 1):
        current = arr[z]
        nxt = arr[z + 1]
        keys_current = [(z, int(v)) for v in np.unique(current) if int(v) > 0]
        keys_next = [(z + 1, int(v)) for v in np.unique(nxt) if int(v) > 0]
        linked_current: dict[tuple[int, int], int] = {}
        linked_next: dict[tuple[int, int], int] = {}

        overlap_keys = _overlap_links(
            current,
            nxt,
            areas,
            min_overlap_fraction=min_overlap_fraction,
            max_area_ratio=max_area_ratio,
            z=z,
        )
        candidate_links = list(overlap_keys)
        if max_centroid_distance is not None:
            candidate_links.extend(
                _centroid_links(
                    keys_current,
                    keys_next,
                    areas,
                    centroids,
                    max_centroid_distance=max_centroid_distance,
                    max_area_ratio=max_area_ratio,
                )
            )

        seen_links: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for a_key, b_key, reason, metric in candidate_links:
            link_key = (a_key, b_key)
            if link_key in seen_links:
                continue
            seen_links.add(link_key)
            if union(nodes[a_key], nodes[b_key]):
                linked_current[a_key] = linked_current.get(a_key, 0) + 1
                linked_next[b_key] = linked_next.get(b_key, 0) + 1
                edges.append(
                    {
                        "z": int(z),
                        "from": int(a_key[1]),
                        "to": int(b_key[1]),
                        "reason": reason,
                        "metric": float(metric),
                    }
                )

        ambiguous_pairs += sum(1 for count in linked_current.values() if count > 1)
        ambiguous_pairs += sum(1 for count in linked_next.values() if count > 1)

    root_to_label: dict[int, int] = {}
    out = np.zeros(arr.shape, dtype=np.int32)
    for key, node_id in nodes.items():
        root = find(node_id)
        label = root_to_label.setdefault(root, len(root_to_label) + 1)
        z, local_id = key
        out[z][arr[z] == local_id] = label

    record = {
        "plane_roi_count": int(len(nodes)),
        "stitched_cell_count": int(len(root_to_label)),
        "edge_count": int(len(edges)),
        "ambiguous_stitch_pairs": int(ambiguous_pairs),
        "min_overlap_fraction": float(min_overlap_fraction),
        "max_centroid_distance": max_centroid_distance,
        "max_area_ratio": float(max_area_ratio),
        "edges": edges[:200],
    }
    return out, record


def build_auto3d_candidates(
    image: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    base_options: dict[str, Any],
    candidate_modes: list[str] | None = None,
    boundary_mask: np.ndarray | None = None,
    max_candidates: int = 8,
    stitch_min_overlap: float = 0.2,
    stitch_max_centroid_distance: float | None = None,
    stitch_max_area_ratio: float = 3.0,
) -> list[SegmentationCandidate]:
    modes = candidate_modes or ["direct_3d", "plane_stitch"]
    variants = _candidate_variants(base_options)
    candidates: list[SegmentationCandidate] = []
    for params in variants:
        if len(candidates) >= max_candidates:
            break
        if "direct_3d" in modes:
            labels, record = segment_target_array(
                image,
                spacing=spacing,
                boundary_mask=boundary_mask,
                **params,
            )
            candidates.append(
                _candidate_from_labels(
                    "direct_3d",
                    "direct_3d",
                    labels,
                    image,
                    params={**params, **_compact_record(record)},
                )
            )
        if len(candidates) >= max_candidates:
            break
        if "plane_stitch" in modes:
            labels, record = segment_target_array_plane_stitch(
                image,
                spacing=spacing,
                boundary_mask=boundary_mask,
                stitch_min_overlap=stitch_min_overlap,
                stitch_max_centroid_distance=stitch_max_centroid_distance,
                stitch_max_area_ratio=stitch_max_area_ratio,
                **params,
            )
            candidates.append(
                _candidate_from_labels(
                    "plane_stitch",
                    "plane_stitch",
                    labels,
                    image,
                    params={**params, **_compact_record(record)},
                    stitch_record=record.get("stitch"),
                )
            )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def rank_segmentation_labels(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    stitch_record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], float]:
    raw = np.asarray(image, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    noise_sigma = robust_background_sigma(raw)
    signal_qc, warnings = target_object_qc(
        raw,
        raw,
        labels_arr,
        noise_sigma=noise_sigma,
    )
    metrics = {
        **label_qc(labels_arr),
        **signal_qc,
        "single_plane_object_fraction": _single_plane_object_fraction(labels_arr),
        "z_gap_object_fraction": _z_gap_object_fraction(labels_arr),
        "tiny_object_fraction": _tiny_object_fraction(labels_arr),
        "largest_to_median_object_ratio": _largest_to_median_ratio(labels_arr),
    }
    if stitch_record:
        metrics["ambiguous_stitch_pairs"] = int(stitch_record.get("ambiguous_stitch_pairs", 0))
        metrics["stitch_edge_count"] = int(stitch_record.get("edge_count", 0))
        metrics["plane_roi_count"] = int(stitch_record.get("plane_roi_count", 0))

    score = 100.0
    n_objects = int(metrics.get("n_objects", 0))
    if n_objects == 0:
        return metrics, warnings + ["candidate produced zero 3D objects"], -1000.0

    mask_fraction = float(metrics.get("mask_fraction", 0.0))
    if mask_fraction < 0.0002:
        score -= 45.0
        warnings.append("candidate mask covers almost none of the stack")
    elif mask_fraction < 0.001:
        score -= 15.0
    if mask_fraction > 0.45:
        score -= 70.0
        warnings.append("candidate mask covers too much of the stack")
    elif mask_fraction > 0.25:
        score -= 25.0

    bright_outside = float(metrics.get("top_bright_outside_fraction", 0.0))
    score -= min(70.0, 80.0 * bright_outside)

    separation = float(metrics.get("inside_corrected_mean", 0.0)) - float(
        metrics.get("outside_corrected_mean", 0.0)
    )
    if noise_sigma > 0:
        separation_snr = separation / noise_sigma
        metrics["inside_outside_separation_snr"] = float(separation_snr)
        if separation_snr < 1.0:
            score -= 30.0
        else:
            score += min(20.0, 3.0 * separation_snr)

    single_plane = float(metrics.get("single_plane_object_fraction", 0.0))
    if labels_arr.ndim == 3 and labels_arr.shape[0] > 1:
        score -= 25.0 * single_plane
        if single_plane > 0.5:
            warnings.append("many objects appear in only one z plane")

    tiny_fraction = float(metrics.get("tiny_object_fraction", 0.0))
    score -= 25.0 * tiny_fraction
    if tiny_fraction > 0.35:
        warnings.append("candidate contains many tiny objects")

    largest_ratio = float(metrics.get("largest_to_median_object_ratio", 1.0))
    if largest_ratio > 1000:
        score -= 80.0
        warnings.append("candidate has an extreme merged object compared with the median")
    elif largest_ratio > 100:
        score -= 55.0
        warnings.append("candidate has a very large merged object compared with the median")
    elif largest_ratio > 20:
        score -= 25.0
    elif largest_ratio > 6:
        score -= 15.0

    gap_fraction = float(metrics.get("z_gap_object_fraction", 0.0))
    score -= 20.0 * gap_fraction

    ambiguous = int(metrics.get("ambiguous_stitch_pairs", 0))
    if ambiguous:
        score -= min(30.0, ambiguous * 4.0)
        warnings.append("plane stitching produced ambiguous one-to-many links")

    return metrics, warnings, float(score)


def selection_confidence(candidates: list[SegmentationCandidate]) -> str:
    if not candidates:
        return "fail"
    best = candidates[0]
    if best.score < 35 or int(best.metrics.get("n_objects", 0)) == 0:
        return "low"
    if len(candidates) == 1:
        return "medium" if best.score >= 60 else "low"
    margin = float(best.score - candidates[1].score)
    if (
        best.score >= 70
        and candidates[1].score >= 70
        and best.strategy == candidates[1].strategy
    ):
        return "high"
    if best.score >= 75 and margin >= 8:
        return "high"
    if best.score >= 55 and margin >= 3:
        return "medium"
    return "low"


def _candidate_from_labels(
    name: str,
    strategy: str,
    labels: np.ndarray,
    image: np.ndarray,
    *,
    params: dict[str, Any],
    stitch_record: dict[str, Any] | None = None,
) -> SegmentationCandidate:
    metrics, warnings, score = rank_segmentation_labels(
        image,
        labels,
        stitch_record=stitch_record,
    )
    return SegmentationCandidate(
        name=name,
        strategy=strategy,
        labels=np.asarray(labels, dtype=np.int32),
        params=params,
        metrics=metrics,
        warnings=warnings,
        score=score,
    )


def _candidate_variants(base_options: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(base_options)
    variants = [base]
    permissive = dict(base)
    permissive["min_snr"] = max(0.5, float(base["min_snr"]) * 0.75)
    permissive["high_snr"] = max(permissive["min_snr"], float(base["high_snr"]) * 0.8)
    if base.get("min_size") is not None:
        permissive["min_size"] = max(1, int(round(float(base["min_size"]) * 0.7)))
    variants.append(permissive)

    conservative = dict(base)
    conservative["min_snr"] = float(base["min_snr"]) * 1.3
    conservative["high_snr"] = float(base["high_snr"]) * 1.2
    if base.get("min_size") is not None:
        conservative["min_size"] = max(1, int(round(float(base["min_size"]) * 1.4)))
    variants.append(conservative)

    smoother = dict(base)
    smoother["smoothing_sigma"] = float(base["smoothing_sigma"]) + 0.5
    variants.append(smoother)

    return variants


def _overlap_links(
    current: np.ndarray,
    nxt: np.ndarray,
    areas: dict[tuple[int, int], int],
    *,
    min_overlap_fraction: float,
    max_area_ratio: float,
    z: int,
) -> list[tuple[tuple[int, int], tuple[int, int], str, float]]:
    mask = (current > 0) & (nxt > 0)
    if not np.any(mask):
        return []
    pairs = np.stack([current[mask], nxt[mask]], axis=1).astype(np.int64)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    links: list[tuple[tuple[int, int], tuple[int, int], str, float]] = []
    for (a_raw, b_raw), count in zip(unique_pairs, counts, strict=False):
        a_key = (int(z), int(a_raw))
        b_key = (int(z + 1), int(b_raw))
        area_a = max(1, areas.get(a_key, 1))
        area_b = max(1, areas.get(b_key, 1))
        area_ratio = max(area_a, area_b) / max(1, min(area_a, area_b))
        overlap_fraction = float(count / max(1, min(area_a, area_b)))
        if overlap_fraction >= min_overlap_fraction and area_ratio <= max_area_ratio:
            links.append((a_key, b_key, "overlap", overlap_fraction))
    return links


def _centroid_links(
    current: list[tuple[int, int]],
    nxt: list[tuple[int, int]],
    areas: dict[tuple[int, int], int],
    centroids: dict[tuple[int, int], tuple[float, float]],
    *,
    max_centroid_distance: float,
    max_area_ratio: float,
) -> list[tuple[tuple[int, int], tuple[int, int], str, float]]:
    links: list[tuple[tuple[int, int], tuple[int, int], str, float]] = []
    for a_key in current:
        ay, ax = centroids[a_key]
        area_a = max(1, areas[a_key])
        for b_key in nxt:
            by, bx = centroids[b_key]
            area_b = max(1, areas[b_key])
            area_ratio = max(area_a, area_b) / max(1, min(area_a, area_b))
            if area_ratio > max_area_ratio:
                continue
            distance = float(np.hypot(ay - by, ax - bx))
            if distance <= max_centroid_distance:
                links.append((a_key, b_key, "centroid", distance))
    return links


def _single_plane_object_fraction(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.ndim != 3:
        return 0.0
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return 0.0
    presence = np.zeros((n + 1, arr.shape[0]), dtype=bool)
    for z in range(arr.shape[0]):
        ids = np.unique(arr[z])
        ids = ids[ids > 0]
        presence[ids, z] = True
    plane_counts = presence[1:].sum(axis=1)
    return float(np.mean(plane_counts <= 1))


def _z_gap_object_fraction(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.ndim != 3:
        return 0.0
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return 0.0
    gap_count = 0
    for label in range(1, n + 1):
        z_indices = np.flatnonzero(np.any(arr == label, axis=(1, 2)))
        if z_indices.size <= 2:
            continue
        if z_indices[-1] - z_indices[0] + 1 != z_indices.size:
            gap_count += 1
    return float(gap_count / n)


def _tiny_object_fraction(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return 0.0
    counts = np.bincount(arr.ravel(), minlength=n + 1)[1:]
    median = float(np.median(counts))
    if median <= 0:
        return 0.0
    return float(np.mean(counts < median * 0.25))


def _largest_to_median_ratio(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    n = int(arr.max()) if arr.size else 0
    if n == 0:
        return 0.0
    counts = np.bincount(arr.ravel(), minlength=n + 1)[1:]
    median = float(np.median(counts))
    if median <= 0:
        return 0.0
    return float(np.max(counts) / median)


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key == "corrected_for_threshold":
            continue
        out[key] = value
    return out


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    from skimage import segmentation

    relabeled, _fw, _inv = segmentation.relabel_sequential(labels)
    return np.asarray(relabeled, dtype=np.int32)
