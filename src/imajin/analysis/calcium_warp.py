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

import os

import numpy as np

from imajin.analysis._numba import lazy_kernel

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


# Below this many warp tasks the process-pool startup isn't worth it (auto mode).
_PARALLEL_WARP_MIN_TASKS = 200


def _warp_frame_task(args):
    """Warp one frame from its landmark set. The transform is (re)built here from
    src/dst, so only arrays cross the process boundary (no Delaunay/tform pickling).
    Returns ``(warped_or_None, valid, reason)``. Module-level so it is picklable."""
    frame, src, dst, min_landmarks, min_density, max_strain, min_angle_deg = args
    from skimage.transform import warp

    q = warp_quality(src, dst, min_landmarks=min_landmarks, min_density=min_density,
                     max_strain=max_strain, min_angle_deg=min_angle_deg)
    if not q["ok"]:
        return None, False, q["reason"]
    return warp(frame, q["tform"], order=1, mode="constant", cval=np.nan), True, "stabilized"


def _run_warp_frames(tasks, n_workers):
    """Run per-frame warp tasks ``[(t, args), ...]`` → ``[(t, warped, valid, reason), ...]``.

    Per-frame warp is CPU-bound and ``skimage.warp`` does not release the GIL, so
    threads make it slower; a fork process pool gives a real speedup instead.
    Runs sequentially for small jobs (auto) or when ``n_workers <= 1``, and always
    falls back to sequential on any pool failure — including non-fork platforms
    (Windows/macOS), where ``get_context('fork')`` raises. So this never hangs or
    breaks stabilization; it only accelerates the Linux/large-movie case.
    """
    if not tasks:
        return []
    if n_workers is None:
        n_workers = (min(8, os.cpu_count() or 1)
                     if len(tasks) >= _PARALLEL_WARP_MIN_TASKS else 1)
    n_workers = int(n_workers)

    def _sequential():
        return [(t, *_warp_frame_task(a)) for t, a in tasks]

    if n_workers <= 1:
        return _sequential()
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("fork")  # raises on non-fork platforms -> fallback
        args_list = [a for _, a in tasks]
        chunk = max(1, len(tasks) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            results = list(ex.map(_warp_frame_task, args_list, chunksize=chunk))
        return [(tasks[i][0], *results[i]) for i in range(len(tasks))]
    except Exception:
        return _sequential()


def dense_stabilize(movie, labels, result, *, min_landmarks=MIN_LANDMARKS,
                    min_density=MIN_DENSITY, max_strain=MAX_STRAIN,
                    min_angle_deg=MIN_ANGLE_DEG, n_workers=None) -> dict:
    movie = np.asarray(movie, float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    base_xy = _centroids_xy(labels)
    lbls = list(base_xy)
    out = movie.copy()
    valid = np.zeros(T, bool)
    reason = np.array(["gated"] * T, dtype=object)

    # Build per-frame warp tasks; pre-gate frames with too few located landmarks
    # (cheap, and keeps them out of the worker pool).
    tasks = []
    for t in range(T):
        use = [l for l in lbls if str(result.reason[l][t]) == "located"]  # direct tracks only
        if len(use) < min_landmarks:
            reason[t] = "too_few_landmarks"
            continue
        src = np.array([base_xy[l] for l in use])                       # (x, y)
        dst = np.array([result.positions[l][t][::-1] for l in use])     # (y,x) -> (x,y)
        tasks.append(
            (t, (movie[t], src, dst, min_landmarks, min_density, max_strain, min_angle_deg))
        )

    for t, warped, ok, rsn in _run_warp_frames(tasks, n_workers):
        reason[t] = rsn
        if ok:
            out[t] = warped
            valid[t] = True
    return {"movie": out, "valid": valid, "reason": reason}


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


def _build_rolling_percentile(njit):
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

    return _kernel


_rolling_percentile_get = lazy_kernel(_build_rolling_percentile)


def _rolling_percentile(inten: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Centered rolling-percentile F0 baseline (NaN-ignoring, edge-truncated).
    numba-accelerated; matches ``_rolling_percentile_numpy`` to fp rounding."""
    fn = _rolling_percentile_get() or _rolling_percentile_numpy
    return fn(np.ascontiguousarray(inten, dtype=np.float64), int(window), float(pct))


def _masked_mean_over_time_numpy(movie, rows, cols, valid):
    """Reference per-frame ROI mean over time: for each valid frame, the mean of
    the fixed ROI pixels (rows, cols), or NaN if any ROI pixel is non-finite (the
    ROI is not fully inside the warped frame). Matches the original inline loop."""
    T = movie.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        if valid[t]:
            patch = movie[t, rows, cols]
            if np.isfinite(patch).all():
                out[t] = float(patch.mean())
    return out


def _build_masked_mean(njit):
    @njit(cache=True)
    def _kernel(movie, rows, cols, valid):
        T = movie.shape[0]
        n = rows.shape[0]
        out = np.full(T, np.nan)
        if n == 0:
            return out
        for t in range(T):
            if not valid[t]:
                continue
            s = 0.0
            finite = True
            for k in range(n):
                v = movie[t, rows[k], cols[k]]
                if not np.isfinite(v):   # ROI not fully inside -> leave NaN
                    finite = False
                    break
                s += v
            if finite:
                out[t] = s / n
        return out

    return _kernel


_masked_mean_get = lazy_kernel(_build_masked_mean)


def _masked_mean_over_time(movie, rows, cols, valid):
    """Per-frame ROI mean over time (numba-accelerated; matches the numpy
    reference to floating-point rounding)."""
    fn = _masked_mean_get() or _masked_mean_over_time_numpy
    return fn(
        np.ascontiguousarray(movie, dtype=np.float64),
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(cols, dtype=np.int64),
        np.ascontiguousarray(valid),
    )


def dense_corrected_dff(stab_movie, labels, valid, *, window=41, pct=10.0) -> dict:
    movie = np.asarray(stab_movie, float)
    labels = np.asarray(labels)
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
        rows, cols = np.nonzero(m)                            # require the FULL core inside each frame
        inten = _masked_mean_over_time(movie, rows, cols, valid)
        f0 = _rolling_percentile(inten, window, pct)
        out[lbl] = (inten - f0) / np.where(f0 != 0, f0, np.nan)
    return out
