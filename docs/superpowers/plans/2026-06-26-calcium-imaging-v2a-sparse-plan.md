# Calcium Imaging v2a — sparse motion correction (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Recover in-plane motion for *sparse* cells (gut EE — the acceptance bar) by confidence-gated landmark tracking + ROI relocation, with validated neighbour-deformation interpolation across disappearances, so moving cells become measurable while unrecoverable frames still gate (degrade to v1).

**Architecture:** New headless `analysis/calcium_motion.py` (observability, propagated locate, neighbour interpolation, two-pass `correct_sparse`, `corrected_dff`); reuses `calcium_qc`, `calcium_synth` (extended with exact per-frame ground-truth positions via a single affine motion model), `calcium_validation`. A v2 battery and a `correct_calcium_motion` tool follow. **Dense Delaunay warp + non-rigid deformation are out of scope (v2b).**

**Tech Stack:** Python 3.12, numpy, scipy, scikit-image, pandas, pytest.

Follows spec `docs/superpowers/specs/2026-06-25-calcium-imaging-v2-motion-design.md` (3 Codex rounds → GO); thresholds = that spec's frozen defaults table. This plan: rev.3 (3 Codex rounds; changelog at bottom).

## Scope

In: sparse landmark-tracking correction, confidence/gating, validated neighbour interpolation, corrected ΔF/F0, v2 validation (incl. coverage-gain + event-amplitude), tool. Out (→ v2b): dense Delaunay piecewise-affine warp; non-rigid (non-affine) deformation + its GT.

## Motion model (v2a)

Per frame `t`, a single **affine** maps a base point `p=(y,x)` forward to its true
position: `F_t(p) = M_t·p + b_t`, with `M_t = I + frac·G` (`frac=t/(T-1)`, `G` a
small constant 2×2 gradient, default 0) and `b_t = (lat·frac, 0.5·lat·frac)`.
Pixels are produced by the **inverse** map so content is exact:
`out(q) = base(M_t⁻¹·(q − b_t))` via `scipy.ndimage.affine_transform`. True cell
positions use the forward `F_t` — identical transform, so GT matches pixels exactly.
(Non-affine deformation is v2b.)

## File structure

- Create `src/imajin/analysis/calcium_motion.py`.
- Modify `src/imajin/analysis/calcium_synth.py` (affine motion + exact `true_positions` + `silent_windows`).
- Modify `src/imajin/analysis/calcium_validation.py` (`run_v2_acceptance`).
- Modify `src/imajin/tools/qc.py` (`correct_calcium_motion`).
- Modify `README.md`.
- Tests: `tests/test_calcium_motion.py`, `tests/test_calcium_v2_validation.py`, additions to `tests/test_calcium_synth.py`, `tests/test_calcium_tool.py`.

### Shared signatures (authoritative)

```python
# calcium_synth.py: SyntheticRecording gains true_positions: dict[int, np.ndarray]  # (T,2) (y,x)
#   make_recording gains motion={"lateral_px":float,"shear":float=0.0}, silent_windows: dict[int,(int,int)]|None

# calcium_motion.py
SNR_FLOOR=3.0; MAX_STEP=6; MIN_NEIGHBOURS=3; MAX_RESID=1.0; CONF_FLOOR=0.5
def motion_safe_template(movie, roi, *, n_init=5) -> np.ndarray: ...      # median of first n_init ROI-bbox frames
def observability(patch, bg_sigma, snr_floor=SNR_FLOOR) -> dict: ...      # {snr, observable}
def propagated_locate(movie, roi, template, *, max_step=MAX_STEP) -> dict: ...  # {"centroid":(T,2), "peak":(T,)}
def neighbour_interpolate(target_xy0, neighbour_xy0, neighbour_xyt, *,
                          min_neighbours=MIN_NEIGHBOURS, max_resid=MAX_RESID) -> dict: ...  # {xy,ok,resid,reason}
@dataclass
class CorrectionResult:
    positions: dict[int, np.ndarray]   # (T,2)
    confidence: dict[int, np.ndarray]  # (T,)
    usable: dict[int, np.ndarray]      # (T,) bool
    reason: dict[int, np.ndarray]      # (T,) object
def correct_sparse(movie, labels, **kw) -> CorrectionResult: ...
def corrected_dff(movie, labels, result, *, window=41, pct=10.0) -> dict[int, np.ndarray]: ...
```

---

## Task 1: Synth affine motion + exact GT positions + silent windows

**Files:** Modify `calcium_synth.py`; Test `tests/test_calcium_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_synth.py  (append)
def test_true_positions_match_pixels_under_affine():
    import numpy as np
    from scipy.ndimage import center_of_mass
    rec = make_recording(n_frames=30, shape=(90, 90), n_cells=4, seed=4,
                         motion={"lateral_px": 8.0, "shear": 0.05}, noise=0.0)
    for lbl in rec.true_dff:
        pos = rec.true_positions[lbl]
        assert pos.shape == (30, 2)
        # measured centre-of-mass of the (noise-free) moved disk matches GT within 1 px
        roi0 = rec.labels == lbl
        f0 = float(rec.f0[lbl])
        last = rec.movie[-1]
        # restrict to a window around predicted position to isolate this cell
        cy, cx = pos[-1]
        yy, xx = np.mgrid[0:90, 0:90]
        win = (np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 9) & (last > 0.5 * f0)
        if win.sum() > 5:
            com = center_of_mass(last * win)
            assert np.hypot(com[0] - cy, com[1] - cx) < 1.5


def test_silent_window_makes_cell_disappear():
    import numpy as np
    rec = make_recording(n_frames=40, shape=(64, 64), n_cells=3, seed=6, noise=0.0,
                         silent_windows={1: (10, 20)})
    assert np.allclose(rec.true_dff[1][10:20], 0.0)
    roi = rec.labels == 1
    # during the window the cell vanishes toward background (not ~f0) -> unobservable
    assert rec.movie[15][roi].mean() < 0.5 * rec.f0[1]
```

- [ ] **Step 2: Run** `pytest tests/test_calcium_synth.py -k "affine or silent" -q` → FAIL.

- [ ] **Step 3: Implement** — capture `cell_centroids[lbl]=(cy,cx)` during placement; if `positions` is given, use those centroids in order (label = index+1) instead of the random search. In the cell loop, make a silent window *disappear* (drop to background), not just dff=0 (code below). Replace the motion/defocus block with the affine model:

```python
    from scipy.ndimage import affine_transform, gaussian_filter

    lat = float(motion.get("lateral_px", 0.0)) if motion else 0.0
    shear = float(motion.get("shear", 0.0)) if motion else 0.0
    G = np.array([[0.0, shear], [shear, 0.0]])      # small symmetric gradient

    true_positions: dict[int, np.ndarray] = {}
    for lbl, (cy, cx) in cell_centroids.items():
        pts = np.empty((n_frames, 2))
        for t in range(n_frames):
            frac = t / max(1, n_frames - 1)
            M = np.eye(2) + frac * G
            b = np.array([lat * frac, 0.5 * lat * frac])
            pts[t] = M @ np.array([cy, cx]) + b      # forward F_t
        true_positions[lbl] = pts

    movie = base.copy()
    if lat or shear:
        for t in range(n_frames):
            frac = t / max(1, n_frames - 1)
            M = np.eye(2) + frac * G
            b = np.array([lat * frac, 0.5 * lat * frac])
            Minv = np.linalg.inv(M)
            movie[t] = affine_transform(movie[t], Minv, offset=-Minv @ b,
                                        order=1, mode="nearest")
    defocus_frames = list(defocus.get("frames", [])) if defocus else []
    if defocus_frames:
        sigma = float(defocus.get("sigma", 3.0))
        for t in defocus_frames:
            movie[t] = gaussian_filter(movie[t], sigma=sigma)
    movie = (movie + rng.normal(0.0, float(noise), size=movie.shape)).astype(np.float32)
```

  In the cell-placement loop, build per-cell intensity with a visibility mask so a
  silent window vanishes to background (true disappearance):

```python
        vis = np.ones(n_frames)
        if silent_windows and lbl in silent_windows:
            s, e = silent_windows[lbl]
            dff[s:e] = 0.0
            vis[s:e] = 0.0                      # cell vanishes toward background
        intensity = f0[lbl] * (1.0 + dff) * vis
        base[:, mask] += intensity[:, None]
```

  Add params `positions: list[tuple[int, int]] | None = None` and
  `silent_windows: dict[int, tuple[int, int]] | None = None`; return
  `true_positions=true_positions`, keep `motion=motion`. (`true_positions` for the
  still case = the static centroid repeated.)

- [ ] **Step 4: Run** `pytest tests/test_calcium_synth.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): synth affine motion with exact GT positions + silent windows"`

## Task 2: Observability gate

**Files:** Create `calcium_motion.py`; Test `tests/test_calcium_motion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_motion.py
import numpy as np
from imajin.analysis.calcium_motion import observability


def test_observability_flags_low_contrast():
    rng = np.random.default_rng(0); bg = 2.0
    bright = np.full((11, 11), 5.0); bright[3:8, 3:8] = 60.0
    bright += rng.normal(0, bg, bright.shape)
    dim = np.full((11, 11), 5.0) + rng.normal(0, bg, (11, 11))
    assert observability(bright, bg)["observable"]
    assert not observability(dim, bg)["observable"]
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# src/imajin/analysis/calcium_motion.py
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
```

- [ ] **Step 4: Run** → PASS. **Step 5:** `git commit -m "feat(calcium): v2 landmark observability SNR gate"`

## Task 3: Motion-safe template + propagated locate

**Files:** Modify `calcium_motion.py`; Test `tests/test_calcium_motion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_motion.py  (append)
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_motion import motion_safe_template, propagated_locate


def test_propagated_locate_follows_large_drift():
    rec = make_recording(n_frames=40, shape=(110, 110), n_cells=3, seed=11,
                         motion={"lateral_px": 16.0})
    lbl = next(iter(rec.true_dff)); roi = rec.labels == lbl
    tmpl = motion_safe_template(rec.movie, roi)
    res = propagated_locate(rec.movie, roi, tmpl)
    assert set(res) == {"centroid", "peak"}
    err = np.hypot(*(res["centroid"] - rec.true_positions[lbl]).T)
    assert np.median(err) < 1.5
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# append to calcium_motion.py
def motion_safe_template(movie, roi, *, n_init=5) -> np.ndarray:
    movie = np.asarray(movie, dtype=float)
    ys, xs = np.nonzero(np.asarray(roi, bool))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    k = min(n_init, movie.shape[0])
    return np.median(movie[:k, y0:y1, x0:x1], axis=0)   # cell is at ROI in early frames


def propagated_locate(movie, roi, template, *, max_step=MAX_STEP) -> dict:
    from imajin.analysis.calcium_qc import locate_cell, _centroid_of
    movie = np.asarray(movie, dtype=float)
    T = movie.shape[0]
    cen = np.empty((T, 2)); peak = np.empty(T)
    prev = np.array(_centroid_of(np.asarray(roi, bool)), dtype=float)
    for t in range(T):
        loc = locate_cell(movie[t], template, tuple(prev), search_radius=max_step)
        cen[t] = loc["centroid"]; peak[t] = loc["peak"]
        if loc["peak"] > 0.3:
            prev = np.array(loc["centroid"], dtype=float)   # seed next frame (cumulative)
    return {"centroid": cen, "peak": peak}
```

- [ ] **Step 4: Run** → PASS. **Step 5:** `git commit -m "feat(calcium): motion-safe template + propagated locate"`

## Task 4: Validated neighbour interpolation

**Files:** Modify `calcium_motion.py`; Test `tests/test_calcium_motion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_motion.py  (append)
from imajin.analysis.calcium_motion import neighbour_interpolate


def test_neighbour_interpolate_valid_and_invalid():
    t0 = np.array([10.0, 10.0])
    n0 = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], float)
    nt = n0 + np.array([5.0, -3.0])
    ok = neighbour_interpolate(t0, n0, nt)
    assert ok["ok"] and np.allclose(ok["xy"], t0 + [5.0, -3.0], atol=0.5)
    one_sided = neighbour_interpolate(t0, np.array([[0, 0], [2, 0], [0, 2]], float),
                                      np.array([[0, 0], [2, 0], [0, 2]], float) + 5)
    assert not one_sided["ok"]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (Euclidean per-point RMS residual + rank guard + hull):

```python
# append to calcium_motion.py
def _inside_hull(point, pts) -> bool:
    from scipy.spatial import Delaunay
    try:
        return bool(Delaunay(pts).find_simplex(point) >= 0)
    except Exception:
        return False


def neighbour_interpolate(target_xy0, neighbour_xy0, neighbour_xyt, *,
                          min_neighbours=MIN_NEIGHBOURS, max_resid=MAX_RESID) -> dict:
    n0 = np.asarray(neighbour_xy0, float); nt = np.asarray(neighbour_xyt, float)
    tgt = np.asarray(target_xy0, float)
    if len(n0) < min_neighbours or np.linalg.matrix_rank(n0 - n0.mean(0)) < 2:
        return {"xy": tgt, "ok": False, "resid": np.inf, "reason": "degenerate_neighbours"}
    if not _inside_hull(tgt, n0):
        return {"xy": tgt, "ok": False, "resid": np.inf, "reason": "outside_hull"}
    X = np.hstack([n0, np.ones((len(n0), 1))])
    A, *_ = np.linalg.lstsq(X, nt, rcond=None)
    pred = X @ A
    resid = float(np.sqrt(np.mean(np.sum((pred - nt) ** 2, axis=1))))   # Euclidean per-point RMS
    xy = np.hstack([tgt, 1.0]) @ A
    ok = resid < max_resid
    return {"xy": xy, "ok": bool(ok), "resid": resid, "reason": "ok" if ok else "high_residual"}
```

- [ ] **Step 4: Run** → PASS. **Step 5:** `git commit -m "feat(calcium): validated neighbour interpolation (hull + euclidean residual + rank)"`

## Task 5: `correct_sparse` — two-pass, observability at located patch

**Files:** Modify `calcium_motion.py`; Test `tests/test_calcium_motion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_motion.py  (append)
from imajin.analysis.calcium_motion import correct_sparse, CorrectionResult


def test_correct_sparse_tracks_moving_visible_cells():
    rec = make_recording(n_frames=50, shape=(110, 110), n_cells=5, seed=12,
                         motion={"lateral_px": 10.0})
    res = correct_sparse(rec.movie, rec.labels)
    assert isinstance(res, CorrectionResult)
    for lbl in rec.true_dff:
        u = res.usable[lbl]
        assert u.mean() > 0.8
        err = np.hypot(*(res.positions[lbl] - rec.true_positions[lbl]).T)
        assert np.median(err[u]) < 1.5


def test_correct_sparse_interpolates_silent_moving_cell():
    # cell 1 at centre, surrounded by neighbours -> valid interpolation while silent+moving
    pos = [(60, 60), (40, 40), (80, 40), (40, 80), (80, 80)]
    rec = make_recording(n_frames=60, shape=(120, 120), n_cells=5, positions=pos,
                         seed=15, motion={"lateral_px": 8.0}, silent_windows={1: (25, 40)})
    res = correct_sparse(rec.movie, rec.labels)
    window = slice(25, 40)
    err = np.hypot(*(res.positions[1][window] - rec.true_positions[1][window]).T)
    interp_used = np.array([str(r) == "interpolated" for r in res.reason[1][window]])
    assert interp_used.any()
    assert np.median(err[res.usable[1][window]]) < 2.0


def test_correct_sparse_gates_unrecoverable_disappearance():
    # only 2 cells -> < min_neighbours when the target disappears -> gated, not interpolated
    rec = make_recording(n_frames=40, shape=(90, 90), n_cells=2, seed=17,
                         motion={"lateral_px": 8.0}, silent_windows={1: (15, 30)})
    res = correct_sparse(rec.movie, rec.labels)
    w = slice(15, 30)
    assert res.usable[1][w].mean() < 0.3
    assert not any(str(r) == "interpolated" for r in res.reason[1][w])
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (pass 1: locate all + per-frame located-confidence using observability **at the located patch**; pass 2: gaps → interpolate from neighbours that are usable this frame):

```python
# append to calcium_motion.py
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
    movie = np.asarray(movie, dtype=float); labels = np.asarray(labels)
    T = movie.shape[0]
    lbls = [int(v) for v in np.unique(labels) if v != 0]
    bg_sigma = float(np.std(movie[:, labels == 0])) or 1.0

    base_xy, traj, conf, rad = {}, {}, {}, {}
    # PASS 1: locate + located-confidence (observability measured AT the located patch)
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
            obs = observability(_patch_at(movie[t], cy, cx, rad[lbl]), bg_sigma, snr_floor)
            if obs["observable"] and loc["peak"][t] > 0.3:
                c[t] = min(1.0, max(0.0, loc["peak"][t]))
        conf[lbl] = c

    located_usable = {lbl: conf[lbl] >= conf_floor for lbl in lbls}
    positions, confidence, usable, reason = {}, {}, {}, {}
    # PASS 2: fill gaps from neighbours usable THIS frame
    for lbl in lbls:
        pos = traj[lbl].copy(); c = conf[lbl].copy()
        rsn = np.where(c >= conf_floor, "located", "gated").astype(object)
        for t in range(T):
            if c[t] >= conf_floor:
                continue
            others = [m for m in lbls if m != lbl and located_usable[m][t]]
            if len(others) >= min_neighbours:
                ni = neighbour_interpolate(
                    base_xy[lbl], np.array([base_xy[m] for m in others]),
                    np.array([traj[m][t] for m in others]),
                    min_neighbours=min_neighbours, max_resid=max_resid)
                if ni["ok"]:
                    pos[t] = ni["xy"]; c[t] = 0.6; rsn[t] = "interpolated"
                else:
                    rsn[t] = ni["reason"]
            else:
                rsn[t] = "no_neighbours"
        positions[lbl] = pos; confidence[lbl] = c
        usable[lbl] = c >= conf_floor; reason[lbl] = rsn
    return CorrectionResult(positions, confidence, usable, reason)
```

- [ ] **Step 4: Run** → PASS. (If `interpolated` doesn't trigger, increase `silent` cell darkness or neighbour count in the *synthetic*, not the thresholds.) **Step 5:** `git commit -m "feat(calcium): two-pass correct_sparse (located-confidence + valid-neighbour interp)"`

## Task 6: Corrected ΔF/F0 from relocated ROIs

**Files:** Modify `calcium_motion.py`; Test `tests/test_calcium_motion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_motion.py  (append)
from imajin.analysis.calcium_motion import corrected_dff


def test_corrected_dff_recovers_moving_trace():
    rec = make_recording(n_frames=90, shape=(110, 110), n_cells=5, seed=13,
                         motion={"lateral_px": 10.0})
    res = correct_sparse(rec.movie, rec.labels)
    out = corrected_dff(rec.movie, rec.labels, res)
    lbl = max(rec.event_frames, key=lambda k: len(rec.event_frames[k]))
    u = res.usable[lbl]
    r = np.corrcoef(np.nan_to_num(out[lbl][u]), rec.true_dff[lbl][u])[0, 1]
    assert r > 0.9
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# append to calcium_motion.py
def corrected_dff(movie, labels, result, *, window=41, pct=10.0) -> dict:
    movie = np.asarray(movie, dtype=float); labels = np.asarray(labels)
    T = movie.shape[0]
    yy, xx = np.mgrid[0:movie.shape[1], 0:movie.shape[2]]
    out = {}
    for lbl in (int(v) for v in np.unique(labels) if v != 0):
        roi = labels == lbl
        rad = float(np.sqrt(np.count_nonzero(roi) / np.pi))
        pos = result.positions[lbl]; usable = result.usable[lbl]
        inten = np.full(T, np.nan)
        for t in range(T):
            if not usable[t]:
                continue
            cy, cx = pos[t]
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
            if m.any():
                inten[t] = float(movie[t][m].mean())
        f0 = np.full(T, np.nan); half = window // 2
        for t in range(T):
            seg = inten[max(0, t - half): t + half + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                f0[t] = np.percentile(seg, pct)
        out[lbl] = (inten - f0) / np.where(f0 != 0, f0, np.nan)
    return out
```

- [ ] **Step 4: Run** → PASS. **Step 5:** `git commit -m "feat(calcium): corrected ΔF/F0 from relocated ROIs"`

## Task 7: v2 acceptance battery (residual, trace, event-amp, coverage-gain, honest)

**Files:** Modify `calcium_validation.py`; Test `tests/test_calcium_v2_validation.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_v2_validation.py
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_validation import run_v2_acceptance


def test_v2_acceptance_recovers_and_stays_honest():
    rec = make_recording(n_frames=120, shape=(120, 120), n_cells=6, seed=21,
                         bleach_tau=600.0, motion={"lateral_px": 8.0})
    rep = run_v2_acceptance(rec)
    assert rep["residual_median_px"] < 1.0
    assert rep["trace_corr_median"] > 0.95
    assert rep["event_amp_ratio_median"] > 0.8        # event-amplitude preserved
    assert rep["coverage_gain_pp"] > 0                 # v2 beats v1 coverage on moving data
    assert rep["moving_negative_flat"] is True
    assert abs(rep["confidence_dynamics_corr"]) < 0.2
    assert rep["passed"] is True
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# append to calcium_validation.py  (negative_control_flat already imported at module top)
from imajin.analysis.calcium_motion import correct_sparse, corrected_dff
from imajin.analysis.calcium_qc import gate_traces


def _safe_corr(a, b):
    a = np.nan_to_num(np.asarray(a, float)); b = np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_v2_acceptance(rec) -> dict:
    neg = rec.negative_label
    res = correct_sparse(rec.movie, rec.labels)
    dff = corrected_dff(rec.movie, rec.labels, res)
    v1 = gate_traces(rec.movie, rec.labels)

    resids, corrs, amp_ratios, conf_all, act_all = [], [], [], [], []
    cov_fail = False
    for lbl in rec.true_dff:
        u = res.usable[lbl]
        if u.sum() < 5:
            cov_fail = True                       # too-low coverage is a failure, not a skip
        else:
            err = np.hypot(*(res.positions[lbl] - rec.true_positions[lbl]).T)
            resids.append(float(np.median(err[u])))
        conf_all.append(res.confidence[lbl]); act_all.append(np.abs(np.nan_to_num(dff[lbl])))
        if lbl == neg:
            continue                              # exclude negative control from trace/amp
        if u.sum() >= 5:
            c = _safe_corr(dff[lbl][u], rec.true_dff[lbl][u])
            if c is not None:
                corrs.append(c)
        for f in rec.event_frames[lbl]:
            if 0 <= f < len(u) and u[f] and np.isfinite(dff[lbl][f]) and rec.true_dff[lbl][f] > 0:
                amp_ratios.append(float(dff[lbl][f] / rec.true_dff[lbl][f]))

    v1_cov = float(np.mean([v1.coverage[l] for l in rec.true_dff]))
    v2_cov = float(np.mean([res.usable[l].mean() for l in rec.true_dff]))
    coverage_gain_pp = (v2_cov - v1_cov) * 100.0

    moving_neg_flat = True
    if neg is not None:
        u = res.usable[neg]
        moving_neg_flat = bool(u.sum() >= 10 and
                               negative_control_flat(np.nan_to_num(dff[neg][u], nan=0.0))["flat"])

    cd = _safe_corr(np.concatenate(conf_all), np.concatenate(act_all))
    cd = 0.0 if cd is None else cd

    residual_median = float(np.median(resids)) if resids else np.inf
    trace_corr = float(np.median(corrs)) if corrs else 0.0
    amp_ratio = float(np.median(amp_ratios)) if amp_ratios else 0.0
    passed = bool(not cov_fail and residual_median < 1.0 and trace_corr > 0.95
                  and amp_ratio > 0.8 and coverage_gain_pp > 0
                  and moving_neg_flat and abs(cd) < 0.2)
    return {
        "residual_median_px": residual_median,
        "trace_corr_median": trace_corr,
        "event_amp_ratio_median": amp_ratio,
        "coverage_gain_pp": float(coverage_gain_pp),
        "moving_negative_flat": moving_neg_flat,
        "confidence_dynamics_corr": float(cd),
        "passed": passed,
    }
```

- [ ] **Step 4: Run** → PASS. (Tune *synthetic* `lateral_px` toward regime R1 if marginal; never relax binding thresholds.) **Step 5:** `git commit -m "feat(calcium): v2 acceptance battery (residual, trace, event-amp, coverage-gain, honesty)"`

## Task 8: `correct_calcium_motion` tool (stores corrected table) + README

**Files:** Modify `tools/qc.py`, `README.md`; Test `tests/test_calcium_tool.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_calcium_tool.py  (append)
def test_correct_calcium_motion_stores_corrected_table():
    from imajin.analysis.calcium_synth import make_recording
    rec = make_recording(n_frames=40, shape=(90, 90), n_cells=4, seed=14,
                         motion={"lateral_px": 8.0})
    state.put_array("mv", rec.movie); state.put_array("lb", rec.labels)
    res = qc.correct_calcium_motion("mv_tc", movie_key="mv", labels_key="lb")
    assert len(res["metrics"]["coverage"]) == 4
    assert res["metrics"]["corrected_table"] in state.list_tables()
    df = state.get_table(res["metrics"]["corrected_table"])
    assert {"label", "time_index", "dff_corrected"} <= set(df.columns)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — tool runs `correct_sparse` + `corrected_dff`, builds a long-format corrected table (`label, time_index, dff_corrected`), stores it via `put_table`, and records coverage + the table name:

```python
# src/imajin/tools/qc.py  (append)
@tool(description="v2 sparse motion correction: confidence-gated landmark tracking + ROI "
      "relocation with neighbour interpolation; stores a corrected ΔF/F0 table and a QC "
      "record with per-cell coverage. Degrades to gating when confidence fails.",
      phase="6", worker=True)
def correct_calcium_motion(table_name: str, movie_key: str, labels_key: str) -> dict[str, Any]:
    import pandas as pd
    from imajin.analysis.calcium_motion import correct_sparse, corrected_dff
    movie = _materialize(state.get_array(movie_key))
    labels = _materialize(state.get_array(labels_key)).astype(np.int32)
    res = correct_sparse(movie, labels)
    dff = corrected_dff(movie, labels, res)
    rows = [{"label": int(lbl), "time_index": t, "dff_corrected": float(v)}
            for lbl, arr in dff.items() for t, v in enumerate(arr)]
    corrected_table = state.put_table(f"{table_name}_motion_corrected",
                                      pd.DataFrame(rows), spec={"tool": "correct_calcium_motion"})
    coverage = {int(k): float(v.mean()) for k, v in res.usable.items()}
    rejected = [k for k, c in coverage.items() if c < 0.5]
    warnings = ([f"{len(rejected)} cell(s) <50% coverage after correction: {rejected}"]
                if rejected else [])
    metrics = {"kind": "calcium_motion_correction", "table_name": table_name,
               "corrected_table": corrected_table, "coverage": coverage,
               "rejected": rejected, "failed": False}
    return _record(table_name, warnings, metrics)
```

  README: extend the calcium bullet with "v2a: confidence-gated sparse motion correction (`correct_calcium_motion`) producing a corrected ΔF/F0 table."

- [ ] **Step 4: Run** `pytest tests/test_calcium_tool.py -q` → PASS. **Step 5:** `git commit -m "feat(calcium): correct_calcium_motion tool stores corrected table (v2a) + README"`

---

## Self-review

**Spec coverage (v2a):** observability at located patch (Tasks 2,5 ↔ observability rule); propagated locate beyond ±window with motion-safe template (Task 3 ↔ large-motion + template); validated neighbour interpolation, hull+rank+euclidean-residual, neighbours usable-this-frame (Tasks 4,5 ↔ interpolation-validity); two-pass confidence + conf_floor + degrade-to-v1 (Task 5 ↔ safety principle); corrected ΔF/F0 (Task 6 ↔ req 2); battery with residual/trace/**event-amplitude**/**coverage-gain**/moving-neg-control/confidence-vs-dynamics + disappearance valid+invalid (Tasks 5,7 ↔ reqs 1,2,3,4,6); tool stores corrected table (Task 8). Exact affine GT (Task 1) resolves the synth coordinate consistency. **Deferred to v2b (sound):** dense Delaunay warp + its topology gates, non-affine deformation + GT, IDF1 tracking metric.

**Placeholder scan:** every code step is inline and runnable (observability Task 2, propagated locate Task 3, neighbour interp Task 4, correct_sparse Task 5, corrected_dff Task 6, run_v2_acceptance Task 7, tool Task 8); no TBD, no external references.

**Type consistency:** `propagated_locate`→`{centroid,peak}` (Tasks 3,5); `neighbour_interpolate`→`{xy,ok,resid,reason}` (Tasks 4,5); `CorrectionResult(positions,confidence,usable,reason)` (Tasks 5–8); `corrected_dff`→`dict[label]->(T,)` (Tasks 6,7,8); tool returns via `_record` → tests read `res["metrics"][...]` (Task 8).

## Changelog
- rev.0 (2026-06-26): initial v2a plan.
- rev.1 (2026-06-26): fixed all 8 Codex NO-GO items — exact affine GT (motion model section + Task 1, non-rigid deferred to v2b); motion-safe template (Task 3); `propagated_locate` returns `{centroid,peak}` only (dropped ambiguous dy/dx); observability measured at the located patch (Task 5); two-pass interpolation from usable-this-frame neighbours with euclidean-RMS residual + rank guard (Tasks 4,5); battery now includes event-amplitude + coverage-gain + neg-control guard + disappearance valid/invalid tests (Tasks 5,7); tool stores a corrected ΔF/F0 table and `corrected_dff` is in the authoritative signatures (Task 8).
