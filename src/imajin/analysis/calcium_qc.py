"""Honest QC gates for calcium ΔF/F0 timecourses (v1, detection-only).

Two independent per-frame gates — defocus (multi-metric, intensity-normalized,
robust) and lateral-motion/ROI-validity — mark each frame usable or not. v1 does
NOT correct motion; it gates and reports coverage so traces are trustworthy or
explicitly rejected. Headless (numpy/scipy only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import laplace, median_filter, shift as nd_shift, sobel


def focus_metrics(patch) -> dict[str, float]:
    """Intensity-normalized sharpness metrics (activity-robust)."""
    p = np.asarray(patch, dtype=np.float64)
    pn = p / (float(np.mean(p)) or 1.0)
    lap = laplace(pn)
    gy, gx = sobel(pn, axis=0), sobel(pn, axis=1)
    std = float(np.std(p)) or 1.0
    snr = (float(np.percentile(p, 90)) - float(np.percentile(p, 10))) / std
    return {"lap_norm": float(np.var(lap)),
            "tenengrad": float(np.mean(gy ** 2 + gx ** 2)),
            "snr": float(snr)}


def _zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x)) or 1.0
    return (x - float(np.mean(x))) / sd


def composite_focus(metrics_over_time: dict[str, np.ndarray]) -> np.ndarray:
    """Per-frame composite = mean of per-metric z-scores over time."""
    return np.mean(np.vstack([_zscore(v) for v in metrics_over_time.values()]), axis=0)


def _locate_cell_reference(frame, template, roi_centroid, search_radius=6) -> dict:
    """Reference (numpy) implementation of :func:`locate_cell` — the exact fallback
    when numba is unavailable and the equivalence reference for the numba kernel."""
    frame = np.asarray(frame, dtype=np.float64)
    tmpl = np.asarray(template, dtype=np.float64)
    tmpl = tmpl - tmpl.mean()
    th, tw = tmpl.shape
    cy, cx = int(round(roi_centroid[0])), int(round(roi_centroid[1]))
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            y0, x0 = cy + dy - th // 2, cx + dx - tw // 2
            if y0 < 0 or x0 < 0 or y0 + th > frame.shape[0] or x0 + tw > frame.shape[1]:
                continue
            win = frame[y0:y0 + th, x0:x0 + tw]
            win = win - win.mean()
            denom = (np.linalg.norm(win) * np.linalg.norm(tmpl)) or 1.0
            candidates.append((float(np.sum(win * tmpl) / denom), dy, dx))
    if not candidates:
        return {"dy": 0, "dx": 0, "peak": -np.inf, "centroid": tuple(roi_centroid)}
    peak = max(c[0] for c in candidates)
    # prefer the smallest displacement among near-best scores (robust to noise ties):
    # a still cell should localize to (0,0), not a noise-driven 1-2 px jitter.
    _, dy, dx = sorted((c for c in candidates if c[0] >= peak - 0.05),
                       key=lambda c: c[1] ** 2 + c[2] ** 2)[0]
    return {"dy": dy, "dx": dx, "peak": peak,
            "centroid": (roi_centroid[0] + dy, roi_centroid[1] + dx)}


_LOCATE_KERNEL = None  # None: not tried yet; False: numba unavailable; else: kernel


def _locate_kernel():
    """Lazily compile + cache the numba normalized-cross-correlation locate kernel
    (compiled on first use so importing this module stays fast). Returns None when
    numba is unavailable, so the caller uses the numpy reference."""
    global _LOCATE_KERNEL
    if _LOCATE_KERNEL is None:
        try:
            from numba import njit

            @njit(cache=True)
            def _kernel(frame, tmpl_c, tmpl_norm, cy, cx, R, th, tw):
                npos = (2 * R + 1) * (2 * R + 1)
                scores = np.full(npos, -1.0e18)
                dys = np.empty(npos, np.int64)
                dxs = np.empty(npos, np.int64)
                H, W = frame.shape
                area = th * tw
                k = 0
                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        dys[k] = dy
                        dxs[k] = dx
                        y0 = cy + dy - th // 2
                        x0 = cx + dx - tw // 2
                        if y0 < 0 or x0 < 0 or y0 + th > H or x0 + tw > W:
                            k += 1
                            continue
                        s = 0.0
                        for i in range(th):
                            for j in range(tw):
                                s += frame[y0 + i, x0 + j]
                        wm = s / area
                        dot = 0.0
                        wn = 0.0
                        for i in range(th):
                            for j in range(tw):
                                wv = frame[y0 + i, x0 + j] - wm
                                dot += wv * tmpl_c[i, j]
                                wn += wv * wv
                        denom = (wn ** 0.5) * tmpl_norm
                        scores[k] = dot / denom if denom > 0 else 0.0
                        k += 1
                peak = scores.max()
                if peak <= -1.0e17:
                    return 0, 0, -np.inf
                best_d = 1 << 60
                bdy = 0
                bdx = 0
                for kk in range(npos):
                    if scores[kk] >= peak - 0.05:
                        d = dys[kk] * dys[kk] + dxs[kk] * dxs[kk]
                        if d < best_d:
                            best_d = d
                            bdy = dys[kk]
                            bdx = dxs[kk]
                return bdy, bdx, peak

            _LOCATE_KERNEL = _kernel
        except Exception:  # pragma: no cover - numba is a declared dependency
            _LOCATE_KERNEL = False
    return _LOCATE_KERNEL or None


def locate_cell(frame, template, roi_centroid, search_radius=6) -> dict:
    """Locate a cell near its ROI by normalized cross-correlation of an
    activity-minimized template. Returns the best (dy, dx), peak score, and the
    shifted centroid. numba-accelerated over the search window (falls back to the
    numpy reference), matching it to floating-point rounding."""
    kernel = _locate_kernel()
    if kernel is None:
        return _locate_cell_reference(frame, template, roi_centroid, search_radius)
    frame = np.ascontiguousarray(frame, dtype=np.float64)
    tmpl = np.asarray(template, dtype=np.float64)
    tmpl_c = np.ascontiguousarray(tmpl - tmpl.mean())
    tmpl_norm = float(np.linalg.norm(tmpl_c))
    th, tw = tmpl_c.shape
    cy, cx = int(round(roi_centroid[0])), int(round(roi_centroid[1]))
    dy, dx, peak = kernel(frame, tmpl_c, tmpl_norm, cy, cx, int(search_radius), th, tw)
    if not np.isfinite(peak):
        return {"dy": 0, "dx": 0, "peak": -np.inf, "centroid": tuple(roi_centroid)}
    return {"dy": int(dy), "dx": int(dx), "peak": float(peak),
            "centroid": (roi_centroid[0] + int(dy), roi_centroid[1] + int(dx))}


def lateral_valid(footprint, roi_mask, centroid, roi_centroid, roi_radius,
                  iou_thresh=0.7, drift_frac=0.5) -> dict:
    if footprint is None:
        return {"iou": 0.0, "drift": np.inf, "ok": False, "reason": "unlocatable"}
    fp = np.asarray(footprint, bool)
    roi = np.asarray(roi_mask, bool)
    inter = float(np.count_nonzero(fp & roi))
    union = float(np.count_nonzero(fp | roi)) or 1.0
    iou = inter / union
    drift = float(np.hypot(centroid[0] - roi_centroid[0], centroid[1] - roi_centroid[1]))
    ok = (iou >= iou_thresh) and (drift < drift_frac * roi_radius)
    reason = "ok" if ok else ("low_iou" if iou < iou_thresh else "drift")
    return {"iou": iou, "drift": drift, "ok": ok, "reason": reason}


@dataclass
class GateResult:
    usable: dict
    coverage: dict
    reason: dict
    longest_run: dict
    missing_frac: dict


def _centroid_of(mask):
    ys, xs = np.nonzero(mask)
    return (float(ys.mean()), float(xs.mean()))


def _longest_true_run(mask) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def gate_traces(movie, labels, *, search_radius=6, iou_thresh=0.7,
                drift_frac=0.5, focus_z=4.0) -> GateResult:
    movie = np.asarray(movie, dtype=np.float64)
    labels = np.asarray(labels)
    T = movie.shape[0]
    usable_d, cov_d, reason_d, run_d, miss_d = {}, {}, {}, {}, {}
    for lbl in (int(v) for v in np.unique(labels) if v != 0):
        roi = labels == lbl
        rcen = _centroid_of(roi)
        rad = float(np.sqrt(np.count_nonzero(roi) / np.pi))
        ys, xs = np.nonzero(roi)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # activity-minimized template = temporal-minimum patch (active frames excluded)
        template = movie[:, y0:y1, x0:x1].min(axis=0)
        metrics = {k: np.empty(T) for k in ("lap_norm", "tenengrad", "snr")}
        for t in range(T):
            fm = focus_metrics(movie[t, y0:y1, x0:x1])
            for k in metrics:
                metrics[k][t] = fm[k]
        comp = composite_focus(metrics)
        # temporal consistency: a single-frame sharpness dip is noise, not defocus
        comp = median_filter(comp, size=3, mode="nearest")
        c_med = float(np.median(comp))
        c_mad = (float(np.median(np.abs(comp - c_med))) * 1.4826) or 1.0
        usable = np.ones(T, bool)
        reason = np.array(["ok"] * T, dtype=object)
        for t in range(T):
            if comp[t] < c_med - focus_z * c_mad:     # robust outlier (minority-defocus)
                usable[t], reason[t] = False, "defocus"
                continue
            loc = locate_cell(movie[t], template, rcen, search_radius)
            # footprint = ROI shifted by the detected displacement (IoU≈1 when still)
            fp = nd_shift(roi.astype(float), (loc["dy"], loc["dx"]),
                          order=0, mode="constant").astype(bool)
            lv = lateral_valid(fp if loc["peak"] > 0.3 else None, roi, loc["centroid"],
                               rcen, rad, iou_thresh, drift_frac)
            if not lv["ok"]:
                usable[t], reason[t] = False, lv["reason"]
        usable_d[lbl] = usable
        cov_d[lbl] = float(usable.mean())
        reason_d[lbl] = reason
        run_d[lbl] = _longest_true_run(usable)
        miss_d[lbl] = float(1.0 - usable.mean())
    return GateResult(usable_d, cov_d, reason_d, run_d, miss_d)
