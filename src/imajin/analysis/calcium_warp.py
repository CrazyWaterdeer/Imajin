"""v2b dense motion correction: landmark-driven piecewise-affine warp.

For dense tissue, stabilise the movie with a Delaunay piecewise-affine warp built
from the directly-located cells (v2a `correct_sparse`), gated by density / triangle
quality / strain / fold checks and **bounded to the landmark convex hull (no
extrapolation)**, then measure fixed ROIs that sit strictly inside the hull.

All warp math is in scikit-image's (x, y) = (col, row) convention; the (y, x)
centroids/positions from `correct_sparse` are swapped before any warp call.
Headless (numpy/scipy/scikit-image).
"""

from __future__ import annotations

import numpy as np

MIN_LANDMARKS = 6
MIN_DENSITY = 1.0 / 2500.0     # >= 1 landmark per (50 px)^2 (spec)
MAX_STRAIN = 1.5
MIN_ANGLE_DEG = 20.0


def _triangle_min_angle(pts) -> float:
    a, b, c = pts
    angs = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        v1, v2 = q - p, r - p
        cos = float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9))
        angs.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return float(min(angs))


def warp_quality(src_xy, dst_xy, *, min_landmarks=MIN_LANDMARKS, min_density=MIN_DENSITY,
                 max_strain=MAX_STRAIN, min_angle_deg=MIN_ANGLE_DEG) -> dict:
    from scipy.spatial import ConvexHull, Delaunay
    from skimage.transform import PiecewiseAffineTransform

    src = np.asarray(src_xy, float)
    dst = np.asarray(dst_xy, float)
    n = len(src)
    if n < min_landmarks:
        return {"ok": False, "reason": "too_few_landmarks", "n": n, "tform": None}
    try:
        area = float(ConvexHull(src).volume)        # 2D hull "volume" == area
    except Exception:
        return {"ok": False, "reason": "degenerate_landmarks", "n": n, "tform": None}
    if area <= 0 or n / area < min_density:
        return {"ok": False, "reason": "low_density", "n": n, "tform": None}
    for simplex in Delaunay(src).simplices:
        if _triangle_min_angle(src[simplex]) < min_angle_deg:
            return {"ok": False, "reason": "thin_triangle", "n": n, "tform": None}
    tform = PiecewiseAffineTransform()
    if not tform.estimate(src, dst):
        return {"ok": False, "reason": "estimate_failed", "n": n, "tform": None}
    for aff in tform.affines:
        lin = np.asarray(aff.params)[:2, :2]
        sv = np.linalg.svd(lin, compute_uv=False)
        if np.linalg.det(lin) <= 0 or sv.max() > max_strain or sv.min() < 1.0 / max_strain:
            return {"ok": False, "reason": "bad_strain_or_fold", "n": n, "tform": None}
    return {"ok": True, "reason": "ok", "n": n, "tform": tform}


def _centroids_xy(labels) -> dict[int, np.ndarray]:
    out = {}
    for lbl in (int(v) for v in np.unique(labels) if v != 0):
        ys, xs = np.nonzero(labels == lbl)
        out[lbl] = np.array([xs.mean(), ys.mean()], float)      # (x, y)
    return out


def interior_labels(labels, *, margin) -> dict[int, np.ndarray]:
    """Labels whose centroid is strictly inside the convex hull of all centroids
    by >= margin (so an ROI of that radius is fully in-hull -> warp defined)."""
    from scipy.spatial import ConvexHull

    cents = _centroids_xy(labels)
    pts = np.array(list(cents.values()))
    hull = ConvexHull(pts)
    eq = hull.equations            # rows [a, b, c]; inside iff a*x + b*y + c <= 0
    out = {}
    for lbl, p in cents.items():
        d = eq[:, :2] @ p + eq[:, 2]
        if np.all(d < -margin):
            out[lbl] = p
    return out


def dense_stabilize(movie, labels, result, *, min_landmarks=MIN_LANDMARKS,
                    min_density=MIN_DENSITY, max_strain=MAX_STRAIN,
                    min_angle_deg=MIN_ANGLE_DEG) -> dict:
    from skimage.transform import warp

    movie = np.asarray(movie, float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    base_xy = _centroids_xy(labels)
    lbls = list(base_xy)
    out = movie.copy()
    valid = np.zeros(T, bool)
    reason = np.array(["gated"] * T, dtype=object)
    for t in range(T):
        use = [l for l in lbls if str(result.reason[l][t]) == "located"]  # direct tracks only
        if len(use) < min_landmarks:
            reason[t] = "too_few_landmarks"
            continue
        src = np.array([base_xy[l] for l in use])                       # (x, y)
        dst = np.array([result.positions[l][t][::-1] for l in use])     # (y,x) -> (x,y)
        q = warp_quality(src, dst, min_landmarks=min_landmarks, min_density=min_density,
                         max_strain=max_strain, min_angle_deg=min_angle_deg)
        if not q["ok"]:
            reason[t] = q["reason"]
            continue
        out[t] = warp(movie[t], q["tform"], order=1, mode="constant", cval=np.nan)
        valid[t] = True
        reason[t] = "stabilized"
    return {"movie": out, "valid": valid, "reason": reason}


_ROLLING_PCT_KERNEL = None


def _rolling_percentile_numpy(inten: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Reference centered rolling-percentile F0 baseline: for each t, the ``pct``
    percentile of the finite values in [t-window//2, t+window//2] (truncated at
    the edges, NaN ignored). The exact numpy fallback and equivalence reference
    for the numba kernel."""
    T = inten.shape[0]
    half = window // 2
    out = np.full(T, np.nan)
    for t in range(T):
        seg = inten[max(0, t - half): t + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[t] = np.percentile(seg, pct)
    return out


def _rolling_percentile_kernel():
    """Lazily compile + cache the numba rolling-percentile kernel (falling back to
    numpy if numba is unavailable). Compiled on first use so importing this module
    stays fast."""
    global _ROLLING_PCT_KERNEL
    if _ROLLING_PCT_KERNEL is not None:
        return _ROLLING_PCT_KERNEL
    try:
        from numba import njit
    except Exception:  # pragma: no cover - numba is a declared dependency
        _ROLLING_PCT_KERNEL = _rolling_percentile_numpy
        return _ROLLING_PCT_KERNEL

    @njit(cache=True)
    def _kernel(inten, window, pct):
        T = inten.shape[0]
        half = window // 2
        out = np.full(T, np.nan)
        buf = np.empty(2 * half + 1, dtype=np.float64)
        for t in range(T):
            lo = t - half if t - half > 0 else 0
            hi = t + half + 1 if t + half + 1 < T else T
            k = 0
            for j in range(lo, hi):
                v = inten[j]
                if not np.isnan(v):
                    buf[k] = v
                    k += 1
            if k == 0:
                continue
            s = np.sort(buf[:k])
            if k == 1:
                out[t] = s[0]
            else:
                # numpy's default 'linear' percentile, replicated exactly.
                rank = (pct / 100.0) * (k - 1)
                lo_i = int(np.floor(rank))
                frac = rank - lo_i
                if lo_i + 1 < k:
                    out[t] = s[lo_i] + frac * (s[lo_i + 1] - s[lo_i])
                else:
                    out[t] = s[lo_i]
        return out

    _ROLLING_PCT_KERNEL = _kernel
    return _ROLLING_PCT_KERNEL


def _rolling_percentile(inten: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Centered rolling-percentile F0 baseline (NaN-ignoring, edge-truncated).
    numba-accelerated; matches ``_rolling_percentile_numpy`` to fp rounding."""
    kernel = _rolling_percentile_kernel()
    return kernel(np.ascontiguousarray(inten, dtype=np.float64), int(window), float(pct))


def dense_corrected_dff(stab_movie, labels, valid, *, window=41, pct=10.0) -> dict:
    movie = np.asarray(stab_movie, float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    valid = np.asarray(valid, bool)
    yy, xx = np.mgrid[0:movie.shape[1], 0:movie.shape[2]]
    radii = {int(v): float(np.sqrt(np.count_nonzero(labels == v) / np.pi))
             for v in np.unique(labels) if v != 0}
    out = {}
    margin = max(2.0, max(radii.values()) - 1.5)
    for lbl, p in interior_labels(labels, margin=margin).items():
        cx, cy = p
        core = max(2.0, radii[lbl] - 1.5)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= core ** 2      # fixed ROI at reference centroid
        inten = np.full(T, np.nan)
        for t in range(T):
            if valid[t]:
                patch = movie[t][m]
                if np.isfinite(patch).all():        # require the FULL core inside this frame's mesh
                    inten[t] = float(np.mean(patch))
        f0 = _rolling_percentile(inten, window, pct)
        out[lbl] = (inten - f0) / np.where(f0 != 0, f0, np.nan)
    return out
