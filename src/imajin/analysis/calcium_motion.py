"""v2a sparse motion correction (detection + correction, confidence-gated).

Recovers in-plane motion for sparse cells by tracking an activity-independent
landmark and relocating the ROI, with validated neighbour-deformation
interpolation across disappearances. Every corrected frame carries a confidence;
below `CONF_FLOOR` the frame is gated (degrade to v1). Headless (numpy/scipy).
Frozen thresholds match the v2 design spec defaults table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SNR_FLOOR = 3.0
MAX_STEP = 6
MIN_NEIGHBOURS = 3
MAX_RESID = 1.0
CONF_FLOOR = 0.5


def observability(patch, bg_sigma, snr_floor=SNR_FLOOR) -> dict:
    p = np.asarray(patch, dtype=float)
    contrast = float(np.percentile(p, 95) - np.percentile(p, 50))
    snr = contrast / (float(bg_sigma) or 1.0)
    return {"snr": snr, "observable": bool(snr >= snr_floor)}


def motion_safe_template(movie, roi, *, n_init=5) -> np.ndarray:
    """Template from the first few frames, where the cell is still at its ROI
    (temporal-min over the whole movie would erase a moving cell)."""
    movie = np.asarray(movie, dtype=float)
    ys, xs = np.nonzero(np.asarray(roi, bool))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    k = min(n_init, movie.shape[0])
    return np.median(movie[:k, y0:y1, x0:x1], axis=0)


def propagated_locate(movie, roi, template, *, max_step=MAX_STEP) -> dict:
    from imajin.analysis.calcium_qc import locate_cell, _centroid_of

    movie = np.asarray(movie, dtype=float)
    T = movie.shape[0]
    cen = np.empty((T, 2))
    peak = np.empty(T)
    prev = np.array(_centroid_of(np.asarray(roi, bool)), dtype=float)
    for t in range(T):
        loc = locate_cell(movie[t], template, tuple(prev), search_radius=max_step)
        cen[t] = loc["centroid"]
        peak[t] = loc["peak"]
        if loc["peak"] > 0.3:
            prev = np.array(loc["centroid"], dtype=float)   # seed next frame (cumulative)
    return {"centroid": cen, "peak": peak}


def _inside_hull(point, pts) -> bool:
    from scipy.spatial import Delaunay

    try:
        return bool(Delaunay(pts).find_simplex(point) >= 0)
    except Exception:
        return False


def neighbour_interpolate(target_xy0, neighbour_xy0, neighbour_xyt, *,
                          min_neighbours=MIN_NEIGHBOURS, max_resid=MAX_RESID) -> dict:
    n0 = np.asarray(neighbour_xy0, float)
    nt = np.asarray(neighbour_xyt, float)
    tgt = np.asarray(target_xy0, float)
    if len(n0) < min_neighbours or np.linalg.matrix_rank(n0 - n0.mean(0)) < 2:
        return {"xy": tgt, "ok": False, "resid": np.inf, "reason": "degenerate_neighbours"}
    if not _inside_hull(tgt, n0):
        return {"xy": tgt, "ok": False, "resid": np.inf, "reason": "outside_hull"}
    X = np.hstack([n0, np.ones((len(n0), 1))])
    A, *_ = np.linalg.lstsq(X, nt, rcond=None)
    resid = float(np.sqrt(np.mean(np.sum((X @ A - nt) ** 2, axis=1))))   # euclidean per-point RMS
    xy = np.hstack([tgt, 1.0]) @ A
    ok = resid < max_resid
    return {"xy": xy, "ok": bool(ok), "resid": resid, "reason": "ok" if ok else "high_residual"}


@dataclass
class CorrectionResult:
    positions: dict
    confidence: dict
    usable: dict
    reason: dict


def _patch_at(movie_t, cy, cx, rad):
    h, w = movie_t.shape
    y0, y1 = max(0, int(cy - rad)), min(h, int(cy + rad) + 1)
    x0, x1 = max(0, int(cx - rad)), min(w, int(cx + rad) + 1)
    return movie_t[y0:y1, x0:x1]


def correct_sparse(movie, labels, *, snr_floor=SNR_FLOOR, max_step=MAX_STEP,
                   min_neighbours=MIN_NEIGHBOURS, max_resid=MAX_RESID,
                   conf_floor=CONF_FLOOR) -> CorrectionResult:
    from imajin.analysis.calcium_qc import _centroid_of

    movie = np.asarray(movie, dtype=float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    lbls = [int(v) for v in np.unique(labels) if v != 0]
    bg_sigma = float(np.std(movie[:, labels == 0])) or 1.0

    base_xy, traj, conf, rad = {}, {}, {}, {}
    # PASS 1: locate + located-confidence (observability AT the located patch)
    for lbl in lbls:
        roi = labels == lbl
        base_xy[lbl] = np.array(_centroid_of(roi), float)
        rad[lbl] = float(np.sqrt(np.count_nonzero(roi) / np.pi))
        tmpl = motion_safe_template(movie, roi)
        loc = propagated_locate(movie, roi, tmpl, max_step=max_step)
        traj[lbl] = loc["centroid"].copy()
        c = np.zeros(T)
        for t in range(T):
            cy, cx = loc["centroid"][t]
            # patch ~2x radius so the cell is a bright minority over local background
            obs = observability(_patch_at(movie[t], cy, cx, 2.0 * rad[lbl]), bg_sigma, snr_floor)
            if obs["observable"] and loc["peak"][t] > 0.3:
                c[t] = min(1.0, max(0.0, loc["peak"][t]))
        conf[lbl] = c

    located_usable = {lbl: conf[lbl] >= conf_floor for lbl in lbls}
    positions, confidence, usable, reason = {}, {}, {}, {}
    # PASS 2: fill gaps from neighbours usable THIS frame
    for lbl in lbls:
        pos = traj[lbl].copy()
        c = conf[lbl].copy()
        rsn = np.where(c >= conf_floor, "located", "gated").astype(object)
        for t in range(T):
            if c[t] >= conf_floor:
                continue
            others = [m for m in lbls if m != lbl and located_usable[m][t]]
            if len(others) >= min_neighbours:
                ni = neighbour_interpolate(
                    base_xy[lbl],
                    np.array([base_xy[m] for m in others]),
                    np.array([traj[m][t] for m in others]),
                    min_neighbours=min_neighbours, max_resid=max_resid,
                )
                if ni["ok"]:
                    pos[t] = ni["xy"]
                    c[t] = 0.6
                    rsn[t] = "interpolated"
                else:
                    rsn[t] = ni["reason"]
            else:
                rsn[t] = "no_neighbours"
        positions[lbl] = pos
        confidence[lbl] = c
        usable[lbl] = c >= conf_floor
        reason[lbl] = rsn
    return CorrectionResult(positions, confidence, usable, reason)


def corrected_dff(movie, labels, result, *, window=41, pct=10.0) -> dict:
    movie = np.asarray(movie, dtype=float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    yy, xx = np.mgrid[0:movie.shape[1], 0:movie.shape[2]]
    out = {}
    for lbl in (int(v) for v in np.unique(labels) if v != 0):
        roi = labels == lbl
        rad = float(np.sqrt(np.count_nonzero(roi) / np.pi))
        core = max(2.0, rad - 1.5)        # sample the cell core: robust to sub-pixel residual
        pos = result.positions[lbl]
        usable = result.usable[lbl]
        inten = np.full(T, np.nan)
        for t in range(T):
            if not usable[t]:
                continue
            cy, cx = pos[t]
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= core ** 2
            if m.any():
                inten[t] = float(movie[t][m].mean())
        f0 = np.full(T, np.nan)
        half = window // 2
        for t in range(T):
            seg = inten[max(0, t - half): t + half + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                f0[t] = np.percentile(seg, pct)
        out[lbl] = (inten - f0) / np.where(f0 != 0, f0, np.nan)
    return out
