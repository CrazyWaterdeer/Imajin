from __future__ import annotations

from typing import Any

import numpy as np

from imajin.analysis.arrays import materialize_array
from imajin.analysis import coords
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return materialize_array(arr)


def _resolve_threshold(arr: np.ndarray, threshold: float | str) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if isinstance(threshold, str):
        if threshold == "otsu":
            from skimage.filters import threshold_otsu

            return float(threshold_otsu(arr))
        if threshold == "zero":
            return 0.0
    raise ValueError(f"unsupported threshold spec: {threshold!r}")


@tool(
    description="Manders' colocalization coefficients M1/M2 between two image layers. "
    "M1 = fraction of channel A intensity that overlaps non-zero channel B. M2 is the "
    "reciprocal. Optionally restrict to a Labels layer mask. Threshold accepts a scalar "
    "or 'otsu' / 'zero'.",
    phase="4",
    worker=True,
)
def manders_coefficients(
    image_a: str,
    image_b: str,
    mask: str | None = None,
    threshold_a: float | str = "otsu",
    threshold_b: float | str = "otsu",
) -> dict[str, Any]:
    a = _materialize(call_on_main(snapshot_layer, image_a).data).astype(np.float64)
    b = _materialize(call_on_main(snapshot_layer, image_b).data).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {image_a} {a.shape} vs {image_b} {b.shape}")

    if mask:
        m = _materialize(call_on_main(snapshot_layer, mask).data) > 0
        if m.shape != a.shape:
            raise ValueError(f"mask shape mismatch: {mask} {m.shape} vs {a.shape}")
    else:
        m = np.ones_like(a, dtype=bool)

    ta = _resolve_threshold(a[m], threshold_a)
    tb = _resolve_threshold(b[m], threshold_b)

    a_in = a[m]
    b_in = b[m]
    a_above = a_in > ta
    b_above = b_in > tb

    sum_a = a_in.sum()
    sum_b = b_in.sum()
    m1 = float(a_in[b_above].sum() / sum_a) if sum_a > 0 else 0.0
    m2 = float(b_in[a_above].sum() / sum_b) if sum_b > 0 else 0.0

    return {
        "M1": m1,
        "M2": m2,
        "threshold_a": ta,
        "threshold_b": tb,
        "n_pixels": int(m.sum()),
        "image_a": image_a,
        "image_b": image_b,
    }


@tool(
    description="Pearson correlation r between two image layers, optionally restricted "
    "to a Labels layer mask. Use when both channels have continuous intensity "
    "distributions (vs Manders for thresholded colocalization).",
    phase="4",
    worker=True,
)
def pearson_correlation(
    image_a: str, image_b: str, mask: str | None = None
) -> dict[str, Any]:
    a = _materialize(call_on_main(snapshot_layer, image_a).data).astype(np.float64)
    b = _materialize(call_on_main(snapshot_layer, image_b).data).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {image_a} {a.shape} vs {image_b} {b.shape}")

    if mask:
        m = _materialize(call_on_main(snapshot_layer, mask).data) > 0
        if m.shape != a.shape:
            raise ValueError(f"mask shape mismatch: {mask} {m.shape} vs {a.shape}")
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()

    if a.size < 2:
        return {"r": 0.0, "n_pixels": int(a.size), "image_a": image_a, "image_b": image_b}

    if a.std() == 0 or b.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(a, b)[0, 1])

    return {
        "r": r,
        "n_pixels": int(a.size),
        "image_a": image_a,
        "image_b": image_b,
    }


# --- Costes automatic threshold + randomization significance -----------------


def _load_pair(image_a: str, image_b: str, mask: str | None):
    a = _materialize(call_on_main(snapshot_layer, image_a).data).astype(np.float64)
    b = _materialize(call_on_main(snapshot_layer, image_b).data).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {image_a} {a.shape} vs {image_b} {b.shape}")
    if mask:
        m = _materialize(call_on_main(snapshot_layer, mask).data) > 0
        if m.shape != a.shape:
            raise ValueError(f"mask shape mismatch: {mask} {m.shape} vs {a.shape}")
    else:
        m = np.ones(a.shape, dtype=bool)
    return a, b, m


def _pearson_vals(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _orthogonal_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Total-least-squares line B = slope*A + intercept (Costes regression)."""
    cov = np.cov(np.vstack([a - a.mean(), b - b.mean()]))
    _w, v = np.linalg.eigh(cov)
    vec = v[:, -1]
    slope = float(vec[1] / vec[0]) if vec[0] != 0 else 0.0
    return slope, float(b.mean() - slope * a.mean())


def _auto_block(img: np.ndarray) -> int:
    """Heuristic block size (px) ~ the spatial autocorrelation half-width, so a
    block preserves sub-PSF structure while randomisation destroys long-range
    correlation. Clamped to [2, 16]; falls back to 3 when flat."""
    x = img - img.mean()
    widths: list[int] = []
    for ax in range(img.ndim):
        line = np.moveaxis(x, ax, 0).reshape(img.shape[ax], -1).mean(axis=1)
        ac = np.correlate(line, line, "full")[len(line) - 1 :]
        if ac.size == 0 or ac[0] <= 0:
            widths.append(3)
            continue
        ac = ac / ac[0]
        below = np.argmax(ac < 0.5)
        widths.append(int(below) if below > 0 else 3)
    return int(np.clip(int(np.mean(widths)) if widths else 3, 2, 16))


def _block_shuffle(img: np.ndarray, block: int, rng) -> np.ndarray:
    """Randomly permute the positions of block x block(x block) tiles. Input
    shape must be divisible by ``block``; output has the same shape."""
    ndim = img.ndim
    grid = tuple(s // block for s in img.shape)
    newshape: list[int] = []
    for g in grid:
        newshape += [g, block]
    x = img.reshape(newshape)
    perm_axes = list(range(0, 2 * ndim, 2)) + list(range(1, 2 * ndim, 2))
    x = np.transpose(x, perm_axes)
    gshape, bshape = x.shape[:ndim], x.shape[ndim:]
    x = x.reshape((-1,) + bshape)
    x = x[rng.permutation(x.shape[0])]
    x = x.reshape(gshape + bshape)
    inv = np.argsort(perm_axes)
    return np.transpose(x, inv).reshape(img.shape)


@tool(
    description="Costes automatic colocalization threshold for two image layers. "
    "Fits the total-least-squares regression between the channels and finds the "
    "intensity thresholds below which the channels are no longer positively "
    "correlated (noise), then reports Manders M1/M2 above those thresholds. More "
    "objective than a fixed Otsu/percentile cutoff. Optionally restrict to a mask. "
    "Sensitive to background/bleedthrough/saturation — provide a specimen mask.",
    phase="4",
    worker=True,
)
def costes_threshold(
    image_a: str, image_b: str, mask: str | None = None
) -> dict[str, Any]:
    a, b, m = _load_pair(image_a, image_b, mask)
    av, bv = a[m], b[m]
    slope, intercept = _orthogonal_fit(av, bv)

    ta = float(av.min())
    for cand in np.percentile(av, np.linspace(99, 1, 50)):
        tb_c = slope * cand + intercept
        sel = (av < cand) & (bv < tb_c)
        if int(sel.sum()) < 3:
            continue
        if _pearson_vals(av[sel], bv[sel]) <= 0:
            ta = float(cand)
            break
    tb = float(slope * ta + intercept)

    a_above, b_above = av > ta, bv > tb
    sum_a, sum_b = av.sum(), bv.sum()
    m1 = float(av[b_above].sum() / sum_a) if sum_a > 0 else 0.0
    m2 = float(bv[a_above].sum() / sum_b) if sum_b > 0 else 0.0
    return {
        "threshold_a": ta,
        "threshold_b": tb,
        "slope": slope,
        "intercept": intercept,
        "M1_above": m1,
        "M2_above": m2,
        "n_pixels": int(m.sum()),
        "image_a": image_a,
        "image_b": image_b,
    }


@tool(
    description="Costes randomization significance for colocalization between two "
    "image layers. Block-scrambles one channel n times (block size ~ the PSF/"
    "autocorrelation width, so sub-resolution structure is preserved) and compares "
    "the observed Pearson r against the null distribution, returning a p-value. "
    "Exploratory — sensitive to background/bleedthrough; use a specimen mask. Raise "
    "n for a finer p-value.",
    phase="4",
    worker=True,
)
def costes_significance(
    image_a: str,
    image_b: str,
    mask: str | None = None,
    n: int = 200,
    block: int | str = "auto",
) -> dict[str, Any]:
    a, b, m = _load_pair(image_a, image_b, mask)
    bsize = _auto_block(a) if block == "auto" else int(block)
    bsize = max(2, min(bsize, min(a.shape)))
    crop = tuple(s - s % bsize for s in a.shape)
    sl = tuple(slice(0, c) for c in crop)
    ac, bc, mc = a[sl], b[sl], m[sl]
    if not mc.any():
        raise ValueError("mask is empty within the analysable (block-cropped) region")

    observed = _pearson_vals(ac[mc], bc[mc])
    rng = np.random.default_rng(0)
    null = np.array(
        [_pearson_vals(ac[mc], _block_shuffle(bc, bsize, rng)[mc]) for _ in range(int(n))]
    )
    p = float((null >= observed).sum() + 1) / (int(n) + 1)  # never reports p=0
    return {
        "observed_r": float(observed),
        "p_value": p,
        "block_size": int(bsize),
        "n_randomizations": int(n),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "significant": bool(p < 0.05 and observed > null.mean()),
        "image_a": image_a,
        "image_b": image_b,
    }


@tool(
    description="Object-based colocalization between two object layers (Points from "
    "detect_spots, or Labels), testing whether objects in A lie closer to objects in "
    "B than expected by chance. Reports the fraction of A within max_distance_um of a "
    "B object, compared to a null model that randomly places the A objects inside the "
    "within_layer specimen mask (count preserved) — so the result reflects proximity, "
    "not object density or segmentation bias. Returns observed fraction, null mean, "
    "p-value, and z-score.",
    phase="4",
    worker=True,
)
def object_colocalization(
    objects_a_layer: str,
    objects_b_layer: str,
    within_layer: str,
    max_distance_um: float = 1.0,
    n: int = 200,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    from imajin.tools.spatial import _extract_objects

    oa = _extract_objects(objects_a_layer)
    ob = _extract_objects(objects_b_layer)
    if not len(oa.centroids) or not len(ob.centroids):
        raise ValueError("both object layers must contain at least one object")

    a_world = coords.data_to_world(oa.centroids, oa.spacing)
    b_world = coords.data_to_world(ob.centroids, ob.spacing)
    tree = cKDTree(b_world)
    d_obs, _ = tree.query(a_world, k=1)
    observed = float((d_obs <= float(max_distance_um)).mean())

    wsnap = call_on_main(snapshot_layer, within_layer)
    within = materialize_array(wsnap.data) > 0
    wspacing = coords.layer_scale(wsnap, within.ndim)
    mask_vox = np.argwhere(within)
    if not len(mask_vox):
        raise ValueError(f"within_layer {within_layer!r} is empty")
    if mask_vox.shape[1] != a_world.shape[1]:
        raise ValueError("within_layer dimensionality must match the object layers")

    rng = np.random.default_rng(0)
    n_a = len(a_world)
    null = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        sample = mask_vox[rng.integers(0, len(mask_vox), size=n_a)]
        sw = coords.data_to_world(sample.astype(float), wspacing)
        dd, _ = tree.query(sw, k=1)
        null[i] = (dd <= float(max_distance_um)).mean()

    p = float((null >= observed).sum() + 1) / (int(n) + 1)
    z = float((observed - null.mean()) / null.std()) if null.std() > 0 else 0.0
    return {
        "observed_fraction": observed,
        "null_mean": float(null.mean()),
        "p_value": p,
        "z_score": z,
        "max_distance_um": float(max_distance_um),
        "n_a": int(n_a),
        "n_b": int(len(b_world)),
        "significant": bool(p < 0.05),
    }
