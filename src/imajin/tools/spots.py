"""Spot / puncta detection — the point-like detection model (synaptic puncta,
FISH spots, vesicles, viral particles).

Multi-scale blob detection (Laplacian-of-Gaussian / Difference-of-Gaussian) on a
target channel, parameterised in **physical µm** (voxel-scale aware, anisotropic
in 3D), with optional background suppression and a hand-drawn boundary. Output is
a napari **Points** layer in data coordinates (canonical geometry frame, see
``analysis/coords.py``) plus a session table carrying both index and µm columns —
so detections flow straight into ``measure`` / ``compare_groups`` / ``plot_*``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis import coords
from imajin.analysis.arrays import materialize_array
from imajin.session import put_qc_record, put_table
from imajin.tools._segmentation_io import resolve_boundary
from imajin.tools.napari_ops import add_points_from_worker, snapshot_layer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return materialize_array(arr, dtype=np.float32)


def _lateral_sigma_um(diameter_um: float, ndim: int) -> float:
    """Characteristic Gaussian sigma (µm) for a blob of the given diameter.

    skimage detects a blob of radius ``r ≈ sqrt(ndim) * sigma``, so
    ``sigma = (diameter / 2) / sqrt(ndim)``.
    """
    return (float(diameter_um) / 2.0) / math.sqrt(ndim)


def _sigma_sequences(
    min_diameter_um: float,
    max_diameter_um: float,
    axial_diameter_um: float | None,
    spacing: tuple[float, ...],
) -> tuple[list[float], list[float]]:
    """Per-axis min/max sigma in **voxels** from physical diameters.

    Lateral (y, x) axes use the lateral diameter range; the axial (z) axis uses
    ``axial_diameter_um`` when given (fixed axial extent) else the lateral range
    — i.e. by default a spot is treated as a physical sphere, and anisotropic
    voxels yield anisotropic voxel-sigma automatically. Real confocal axial PSF
    is elongated, so pass ``axial_diameter_um`` (or raise ``max_diameter_um``)
    for true 3D detection.
    """
    ndim = len(spacing)
    lat_min = _lateral_sigma_um(min_diameter_um, ndim)
    lat_max = _lateral_sigma_um(max_diameter_um, ndim)
    if ndim == 2:
        sy, sx = spacing
        return [lat_min / sy, lat_min / sx], [lat_max / sy, lat_max / sx]
    sz, sy, sx = spacing
    if axial_diameter_um:
        ax = _lateral_sigma_um(axial_diameter_um, ndim)
        min_sz = max_sz = ax / sz
    else:
        min_sz, max_sz = lat_min / sz, lat_max / sz
    return (
        [min_sz, lat_min / sy, lat_min / sx],
        [max_sz, lat_max / sy, lat_max / sx],
    )


def _subtract_background(img: np.ndarray, radius_vox: int) -> np.ndarray:
    """White top-hat: remove background varying more slowly than a spot."""
    from skimage.morphology import ball, disk, white_tophat

    radius_vox = int(max(1, min(radius_vox, 15)))
    footprint = disk(radius_vox) if img.ndim == 2 else ball(radius_vox)
    return white_tophat(img, footprint=footprint)


def _robust_noise(img: np.ndarray) -> tuple[float, float]:
    """(median, MAD-based sigma) of the image, robust to bright spots."""
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    return med, 1.4826 * mad if mad > 0 else float(img.std() or 1.0)


def _sample_window_mean(img: np.ndarray, index: np.ndarray) -> float:
    """Mean intensity in a radius-1 window around a (rounded) voxel index."""
    idx = np.rint(index).astype(int)
    slices = tuple(
        slice(max(0, i - 1), min(s, i + 2)) for i, s in zip(idx, img.shape)
    )
    window = img[slices]
    return float(window.mean()) if window.size else float("nan")


@tool(
    description="Detect point-like spots / puncta (synaptic puncta, FISH spots, "
    "vesicles) on a channel via multi-scale blob detection. Diameters are in µm "
    "(voxel-scale aware, anisotropic in 3D). mode '2d_projection' detects on the "
    "max-projection of a z-stack (z localised by argmax) vs '3d' volumetric "
    "detection. Optional white-tophat background suppression and a boundary_mask "
    "(draw an ROI, run boundary_mask_from_shapes). Adds a Points layer and a "
    "table with subpixel coordinates, estimated diameter, per-channel intensity, "
    "and an SNR/quality score for filtering with filter_table.",
    phase="2",
    vision_hint=True,
    worker=True,
)
def detect_spots(
    channel_layer: str,
    min_diameter_um: float,
    max_diameter_um: float,
    mode: str = "2d_projection",
    method: str = "log",
    threshold_rel: float = 0.1,
    axial_diameter_um: float | None = None,
    subtract_background: bool = True,
    boundary_mask: str | None = None,
    intensity_layers: list[str] | None = None,
    exclude_border: bool = True,
    overlap: float = 0.5,
    num_sigma: int = 5,
    points_layer_name: str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    from skimage.feature import blob_dog, blob_log

    if min_diameter_um <= 0 or max_diameter_um < min_diameter_um:
        raise ValueError("require 0 < min_diameter_um <= max_diameter_um")
    if method not in {"log", "dog"}:
        raise ValueError("method must be 'log' or 'dog'")
    if mode not in {"2d_projection", "3d"}:
        raise ValueError("mode must be '2d_projection' or '3d'")

    snap = call_on_main(snapshot_layer, channel_layer)
    raw = _materialize(snap.data)
    if raw.ndim not in (2, 3):
        raise ValueError(f"detect_spots expects a 2D or 3D layer; got shape {raw.shape}")
    full_spacing = coords.layer_scale(snap, raw.ndim)

    # Resolve the working image + its spacing + the "3D detection?" flag.
    z_argmax = None
    if raw.ndim == 3 and mode == "2d_projection":
        z_argmax = np.argmax(raw, axis=0)
        work = raw.max(axis=0)
        spacing = full_spacing[1:]
        detection_mode = "2d_projection"
    elif raw.ndim == 3:
        work, spacing, detection_mode = raw, full_spacing, "3d"
    else:
        work, spacing, detection_mode = raw, full_spacing, "2d"

    # Background suppression, then normalise to [0, 1] for a scale-free threshold.
    raw_med, raw_noise = _robust_noise(work)
    proc = work.astype(np.float32)
    if subtract_background:
        radius_vox = int(round((max_diameter_um / 2.0) / min(spacing)))
        proc = _subtract_background(proc, radius_vox)
    peak = float(proc.max())
    proc_n = proc / peak if peak > 0 else proc

    min_sigma, max_sigma = _sigma_sequences(
        min_diameter_um, max_diameter_um, axial_diameter_um, spacing
    )

    if method == "log":
        result = blob_log(
            proc_n,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=int(max(1, num_sigma)),
            threshold_rel=float(threshold_rel),
            overlap=float(overlap),
            exclude_border=bool(exclude_border),
        )
    else:
        result = blob_dog(
            proc_n,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold_rel=float(threshold_rel),
            overlap=float(overlap),
            exclude_border=bool(exclude_border),
        )

    wd = work.ndim
    if result.size == 0:
        centers2d = np.empty((0, wd), dtype=float)
        sigmas = np.empty((0, wd), dtype=float)
    else:
        centers2d = result[:, :wd]
        sigmas = result[:, wd : wd + wd]

    # Detection-frame coords (used to index the working image ``proc_n``) vs
    # output-frame coords (lifted into the volume for a 2D projection). They
    # differ only in projection mode.
    det_centers = centers2d
    if z_argmax is not None:
        if len(centers2d):
            yx = np.rint(centers2d).astype(int)
            zc = z_argmax[yx[:, 0], yx[:, 1]].astype(float)
            centers = np.column_stack([zc, centers2d])
            sigmas = np.column_stack([np.zeros(len(sigmas)), sigmas])
        else:
            centers = np.empty((0, 3), dtype=float)
            sigmas = np.empty((0, 3), dtype=float)
        out_spacing = full_spacing
    else:
        centers, out_spacing = centers2d, spacing

    # Restrict to a hand-drawn boundary (filter, not zero, to avoid edge artefacts).
    bmask, braw = resolve_boundary(boundary_mask, raw.shape)
    if bmask is not None and len(centers):
        vox = np.rint(centers).astype(int)
        inside = np.ones(len(centers), dtype=bool)
        for a in range(centers.shape[1]):
            inside &= (vox[:, a] >= 0) & (vox[:, a] < bmask.shape[a])
        keep = inside.copy()
        keep[inside] = bmask[tuple(vox[inside].T)]
        centers, sigmas, det_centers = centers[keep], sigmas[keep], det_centers[keep]

    n = len(centers)
    ndim_out = centers.shape[1] if n else raw.ndim
    axis_names = ("z", "y", "x") if ndim_out == 3 else ("y", "x")

    # Intensity channels: the detection channel plus any extras, sampled on raw data.
    intensity_arrays: dict[str, np.ndarray] = {snap.name: raw}
    for name in intensity_layers or []:
        if name == snap.name:
            continue
        s = call_on_main(snapshot_layer, name)
        arr = _materialize(s.data)
        if arr.shape == raw.shape:
            intensity_arrays[name] = arr

    world = coords.data_to_world(centers, out_spacing) if n else np.empty((0, ndim_out))
    rows: dict[str, Any] = {"spot_id": np.arange(n, dtype=int)}
    for i, ax in enumerate(axis_names):
        rows[ax] = centers[:, i] if n else np.array([])
        rows[f"{ax}_um"] = world[:, i] if n else np.array([])
    # Estimated diameter per axis: d = 2 * sqrt(ndim) * sigma_vox * spacing.
    if n:
        diam = 2.0 * math.sqrt(ndim_out) * sigmas * np.asarray(out_spacing)
        lateral = diam[:, 1:].mean(axis=1) if ndim_out == 3 else diam.mean(axis=1)
        rows["diameter_um"] = lateral
        if ndim_out == 3:
            rows["axial_diameter_um"] = diam[:, 0]
        for name, arr in intensity_arrays.items():
            rows[f"intensity_{name}"] = np.array(
                [_sample_window_mean(arr, centers[k]) for k in range(n)]
            )
        peak_int = rows[f"intensity_{snap.name}"]
        rows["snr"] = (peak_int - raw_med) / raw_noise
        rows["quality"] = np.array(
            [proc_n[tuple(np.rint(det_centers[k]).astype(int))] for k in range(n)]
        )
    else:
        rows["diameter_um"] = np.array([])
        rows["snr"] = np.array([])
        rows["quality"] = np.array([])
    df = pd.DataFrame(rows)

    tname = table_name or f"{snap.name}_spots"
    spec = {
        "op": "detect_spots",
        "channel_layer": snap.name,
        "detection_mode": detection_mode,
        "method": method,
        "min_diameter_um": float(min_diameter_um),
        "max_diameter_um": float(max_diameter_um),
        "axial_diameter_um": axial_diameter_um,
        "threshold_rel": float(threshold_rel),
        "subtract_background": bool(subtract_background),
        "boundary_mask": boundary_mask,
    }
    stored_table = call_on_main(put_table, tname, df, spec=spec)

    pname = points_layer_name or f"{snap.name}_spots"
    display_size = float(df["diameter_um"].median() / max(out_spacing)) if n else 5.0
    call_on_main(
        add_points_from_worker,
        centers if n else np.empty((0, ndim_out)),
        name=pname,
        scale=out_spacing,
        metadata={
            "op": "detect_spots",
            "source_layer": snap.name,
            "table_name": stored_table,
            "detection_mode": detection_mode,
        },
        size=max(2.0, display_size),
        face_color="transparent",
        border_color="yellow",
    )

    # QC: count, density in the analysed field (or boundary), quality spread.
    field_vox = int(braw.astype(bool).sum()) if braw is not None else int(work.size)
    vox_um = float(np.prod(out_spacing))
    unit = "um3" if ndim_out == 3 else "um2"
    density = n / (field_vox * vox_um) if field_vox else 0.0
    warnings: list[str] = []
    if n == 0:
        warnings.append("no spots detected — lower threshold_rel or widen the diameter range")
    if detection_mode == "2d_projection" and raw.ndim == 3:
        warnings.append("z localised by argmax on the projection, not true 3D detection")
    status = "warning" if warnings else "pass"
    metrics = {
        "kind": "spots",
        "n_spots": int(n),
        "detection_mode": detection_mode,
        f"density_per_{unit}": float(density),
        "median_quality": float(df["quality"].median()) if n else 0.0,
        "median_snr": float(df["snr"].median()) if n else 0.0,
    }
    call_on_main(put_qc_record, pname, status, warnings, metrics)

    return {
        "points_layer": pname,
        "table_name": stored_table,
        "n_spots": int(n),
        "detection_mode": detection_mode,
        "density": float(density),
        "density_unit": f"per_{unit}",
        "spacing": tuple(float(v) for v in out_spacing),
        "warnings": warnings,
    }


@tool(
    description="Quality-control a spots table: object count, spatial density, and "
    "the fraction below a quality threshold, with pass/warning/fail status surfaced "
    "in the QC dock. Provide the spots table name from detect_spots.",
    phase="2",
    worker=True,
)
def compute_spots_qc(
    table_name: str,
    min_count: int = 1,
    min_quality: float = 0.0,
) -> dict[str, Any]:
    from imajin.session import get_table

    df = call_on_main(get_table, table_name)
    n = int(len(df))
    below = (
        int((df["quality"] < float(min_quality)).sum())
        if "quality" in df.columns and n
        else 0
    )
    frac_below = below / n if n else 0.0
    warnings: list[str] = []
    if n < int(min_count):
        warnings.append(f"only {n} spots (< min_count {min_count})")
    if frac_below > 0.5:
        warnings.append(f"{frac_below:.0%} of spots below quality {min_quality}")
    status = "fail" if n < int(min_count) else ("warning" if warnings else "pass")
    metrics = {
        "kind": "spots_qc",
        "n_spots": n,
        "n_below_quality": below,
        "fraction_below_quality": float(frac_below),
    }
    call_on_main(put_qc_record, table_name, status, warnings, metrics)
    return {"table_name": table_name, "status": status, "n_spots": n, "warnings": warnings}
