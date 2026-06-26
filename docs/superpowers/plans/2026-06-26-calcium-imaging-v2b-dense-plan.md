# Calcium Imaging v2b — dense warp motion correction (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** For *dense* tissue (epithelial sheets), stabilise the movie with a landmark-driven Delaunay **piecewise-affine warp** — gated by density/triangle/strain/fold checks and **bounded to the landmark convex hull (no extrapolation)** — then measure fixed ROIs that lie strictly inside the hull on the stabilised footage.

**Architecture:** New headless `analysis/calcium_warp.py`. Landmarks = the confidently-tracked cells from v2a `correct_sparse` (reuse). All warp math is in scikit-image's **(x, y) = (col, row)** convention (label centroids and `positions`, which are (y, x), are swapped once). Per frame: `PiecewiseAffineTransform.estimate(src=reference, dst=current)`, validate that exact transform (`warp_quality`), then `warp(frame, tform, mode="constant", cval=nan)` (pixels outside the mesh become NaN). Measurement is restricted to cells strictly interior to the reference hull. Frames failing the gates are marked invalid (degrade to v1/gate).

**Tech Stack:** Python 3.12, numpy, scipy (`ConvexHull`, `Delaunay`), scikit-image (`PiecewiseAffineTransform`, `warp`), pytest.

Follows the v2 spec (dense path + density/triangle/strain/fold gates + "no extrapolation beyond the hull"). Reuses v2a `correct_sparse`. This plan: rev.2 (2 Codex rounds; changelog at bottom).

## Coordinate & hull conventions (precise)

- **(x, y) = (col, row).** `correct_sparse` returns `positions` and label centroids as (y, x); swap with `[..., ::-1]` before any warp call.
- **Direction:** `tform = PiecewiseAffineTransform.estimate(src=reference_xy, dst=current_xy)`; `warp(frame_t, tform)` computes `out[c] = frame_t[tform(c)]`, so `out[reference] = frame_t[current]` → moved content pulled back to the reference (stabilised).
- **No extrapolation:** `warp(..., mode="constant", cval=np.nan)`; pixels outside the mesh are NaN. A cell is **measurable only if its reference centroid is strictly inside the reference convex hull by a margin ≥ its ROI core radius** (so the whole ROI is in-hull). Hull-edge cells are skipped.

## File structure

- Create `src/imajin/analysis/calcium_warp.py` — `warp_quality`, `interior_labels`, `dense_stabilize`, `dense_corrected_dff`.
- Modify `src/imajin/analysis/calcium_validation.py` — `run_v2b_acceptance`.
- Modify `src/imajin/tools/qc.py` — `stabilize_calcium_dense` tool.
- Modify `README.md`.
- Tests: `tests/test_calcium_warp.py`, `tests/test_calcium_v2b_validation.py`, additions to `tests/test_calcium_tool.py`.

### Shared signatures (authoritative; all points (x, y))

```python
MIN_LANDMARKS = 6
MIN_DENSITY = 1.0 / 2500.0     # >= 1 landmark per (50 px)^2 (spec)
MAX_STRAIN = 1.5
MIN_ANGLE_DEG = 20.0

def warp_quality(src_xy, dst_xy, *, min_landmarks=MIN_LANDMARKS, min_density=MIN_DENSITY,
                 max_strain=MAX_STRAIN, min_angle_deg=MIN_ANGLE_DEG) -> dict: ...   # {ok,reason,n,tform}
def interior_labels(labels, *, margin) -> dict[int, np.ndarray]: ...   # label -> (x,y) centroid, strictly in-hull
def dense_stabilize(movie, labels, result, **gates) -> dict: ...       # {movie(NaN out-of-hull), valid, reason}
def dense_corrected_dff(stab_movie, labels, valid, *, window=41, pct=10.0) -> dict[int, np.ndarray]: ...
```

### Shared test layout (interior cells exist)

```python
# 4x4 grid in a 160x160 FOV; first 12 = perimeter (hull), last 4 = interior 2x2.
# negative_label = n_cells = 16 -> an interior cell -> measurable & flat.
PERIM = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
         (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135)]
INNER = [(65, 65), (65, 100), (100, 65), (100, 100)]   # labels 13,14,15,16(=neg)
POS16 = PERIM + INNER
```

---

## Task 1: `warp_quality` (density / triangle / strain / fold) + `interior_labels`

**Files:** Create `src/imajin/analysis/calcium_warp.py`; Test `tests/test_calcium_warp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_warp.py
import numpy as np
from imajin.analysis.calcium_warp import warp_quality, interior_labels


def _grid_xy(n=4, step=35, origin=30):
    return np.array([(origin + j * step, origin + i * step)      # (x, y)
                     for i in range(n) for j in range(n)], float)


def test_warp_quality_accepts_mild_and_rejects_fold_and_sparse():
    src = _grid_xy()
    assert warp_quality(src, src + np.array([3.0, -2.0]))["ok"]          # translation
    folded = src.copy(); folded[5] = folded[5] + np.array([55.0, 55.0])  # yank interior pt
    assert not warp_quality(src, folded)["ok"]
    assert not warp_quality(src[:4], src[:4])["ok"]                      # < MIN_LANDMARKS


def test_interior_labels_excludes_hull_cells():
    labels = np.zeros((160, 160), np.int32)
    POS16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
             (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
             (65, 65), (65, 100), (100, 65), (100, 100)]
    yy, xx = np.mgrid[0:160, 0:160]
    for i, (cy, cx) in enumerate(POS16, start=1):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25] = i
    inner = interior_labels(labels, margin=4.0)
    assert set(inner) == {13, 14, 15, 16}        # only the inner 2x2 are strictly in-hull
```

- [ ] **Step 2: Run** `pytest tests/test_calcium_warp.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/imajin/analysis/calcium_warp.py
from __future__ import annotations

import numpy as np

MIN_LANDMARKS = 6
MIN_DENSITY = 1.0 / 2500.0
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
```

- [ ] **Step 4: Run** `pytest tests/test_calcium_warp.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): v2b warp_quality (density/triangle/strain/fold) + interior_labels"`

## Task 2: `dense_stabilize` — gated warp, (x,y), NaN outside hull

**Files:** Modify `calcium_warp.py`; Test `tests/test_calcium_warp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_warp.py  (append)
from scipy.ndimage import center_of_mass
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_motion import correct_sparse
from imajin.analysis.calcium_warp import dense_stabilize

POS16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
         (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
         (65, 65), (65, 100), (100, 65), (100, 100)]


def test_dense_stabilize_pulls_interior_cell_back_on_a_moving_frame():
    rec = make_recording(n_frames=30, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=31, motion={"lateral_px": 6.0}, noise=0.0)
    res = correct_sparse(rec.movie, rec.labels)
    stab = dense_stabilize(rec.movie, rec.labels, res)
    assert stab["valid"].mean() > 0.8
    t = int(np.where(stab["valid"])[0][-1])          # a LATE valid frame (motion present)
    base = np.array(center_of_mass(rec.labels == 13))   # interior cell, (row,col)
    f0 = float(rec.f0[13])
    frame = np.nan_to_num(stab["movie"][t], nan=0.0)
    yy, xx = np.mgrid[0:160, 0:160]
    win = (np.abs(yy - base[0]) < 7) & (np.abs(xx - base[1]) < 7) & (frame > 0.5 * f0)
    com = np.array(center_of_mass(frame * win))
    assert np.hypot(*(com - base)) < 2.0
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# append to calcium_warp.py
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
        use = [l for l in lbls if str(result.reason[l][t]) == "located"]  # direct tracks only (no interpolated)
        if len(use) < min_landmarks:
            reason[t] = "too_few_landmarks"
            continue
        src = np.array([base_xy[l] for l in use])                       # (x, y)
        dst = np.array([result.positions[l][t][::-1] for l in use])     # (y,x)->(x,y)
        q = warp_quality(src, dst, min_landmarks=min_landmarks, min_density=min_density,
                         max_strain=max_strain, min_angle_deg=min_angle_deg)
        if not q["ok"]:
            reason[t] = q["reason"]
            continue
        out[t] = warp(movie[t], q["tform"], order=1, mode="constant", cval=np.nan)
        valid[t] = True
        reason[t] = "stabilized"
    return {"movie": out, "valid": valid, "reason": reason}
```

- [ ] **Step 4: Run** `pytest tests/test_calcium_warp.py -q` → PASS. (If `valid.mean()` is marginal, widen the grid / lower motion in the *synthetic*; never relax the gates.)

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): v2b dense_stabilize (gated (x,y) warp, NaN outside hull)"`

## Task 3: `dense_corrected_dff` — interior fixed ROIs on stabilised movie

**Files:** Modify `calcium_warp.py`; Test `tests/test_calcium_warp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_warp.py  (append)
from imajin.analysis.calcium_warp import dense_corrected_dff


def test_dense_corrected_dff_recovers_interior_trace():
    rec = make_recording(n_frames=90, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=32, motion={"lateral_px": 6.0})
    res = correct_sparse(rec.movie, rec.labels)
    stab = dense_stabilize(rec.movie, rec.labels, res)
    out = dense_corrected_dff(stab["movie"], rec.labels, stab["valid"])
    assert set(out) == {13, 14, 15, 16}              # only interior cells measured
    # an interior signalling cell (not the neg control 16) recovers its trace
    lbl = max((k for k in (13, 14, 15) if rec.event_frames[k]),
              key=lambda k: len(rec.event_frames[k]))
    v = stab["valid"]
    r = np.corrcoef(np.nan_to_num(out[lbl][v]), rec.true_dff[lbl][v])[0, 1]
    assert r > 0.95
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# append to calcium_warp.py
def dense_corrected_dff(stab_movie, labels, valid, *, window=41, pct=10.0) -> dict:
    movie = np.asarray(stab_movie, float)
    labels = np.asarray(labels)
    T = movie.shape[0]
    valid = np.asarray(valid, bool)
    yy, xx = np.mgrid[0:movie.shape[1], 0:movie.shape[2]]
    radii = {int(v): float(np.sqrt(np.count_nonzero(labels == v) / np.pi))
             for v in np.unique(labels) if v != 0}
    out = {}
    for lbl, p in interior_labels(labels, margin=max(2.0, max(radii.values()) - 1.5)).items():
        cx, cy = p
        core = max(2.0, radii[lbl] - 1.5)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= core ** 2      # fixed ROI at reference centroid
        inten = np.full(T, np.nan)
        for t in range(T):
            if valid[t]:
                patch = movie[t][m]
                if np.isfinite(patch).all():        # require the FULL core inside this frame's mesh
                    inten[t] = float(np.mean(patch))
        f0 = np.full(T, np.nan)
        half = window // 2
        for t in range(T):
            seg = inten[max(0, t - half): t + half + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                f0[t] = np.percentile(seg, pct)
        out[lbl] = (inten - f0) / np.where(f0 != 0, f0, np.nan)
    return out
```

- [ ] **Step 4: Run** `pytest tests/test_calcium_warp.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): v2b dense_corrected_dff (interior fixed ROIs on stabilised movie)"`

## Task 4: v2b acceptance battery

**Files:** Modify `calcium_validation.py`; Test `tests/test_calcium_v2b_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_v2b_validation.py
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_validation import run_v2b_acceptance

POS16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
         (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
         (65, 65), (65, 100), (100, 65), (100, 100)]


def test_v2b_acceptance_recovers_and_stays_honest():
    rec = make_recording(n_frames=120, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=41, bleach_tau=600.0, motion={"lateral_px": 6.0})
    rep = run_v2b_acceptance(rec)
    assert rep["valid_fraction"] > 0.8
    assert rep["trace_corr_median"] > 0.95
    assert rep["moving_negative_flat"] is True
    assert rep["passed"] is True


def test_v2b_gates_when_under_constrained():
    rec = make_recording(n_frames=30, shape=(90, 90), n_cells=3, seed=42,
                         motion={"lateral_px": 6.0})       # < MIN_LANDMARKS
    rep = run_v2b_acceptance(rec)
    assert rep["valid_fraction"] < 0.2
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — reuse `_safe_corr` and `negative_control_flat` already in the module; interior cells only (whatever `dense_corrected_dff` returns); neg flatness only if neg is among the interior (measured) cells:

```python
# append to calcium_validation.py
def run_v2b_acceptance(rec) -> dict:
    from imajin.analysis.calcium_motion import correct_sparse
    from imajin.analysis.calcium_warp import dense_stabilize, dense_corrected_dff

    neg = rec.negative_label
    res = correct_sparse(rec.movie, rec.labels)
    stab = dense_stabilize(rec.movie, rec.labels, res)
    dff = dense_corrected_dff(stab["movie"], rec.labels, stab["valid"])
    v = stab["valid"]

    corrs = []
    for lbl in dff:
        if lbl == neg or v.sum() < 5:
            continue
        c = _safe_corr(dff[lbl][v], rec.true_dff[lbl][v])
        if c is not None:
            corrs.append(c)

    moving_neg_flat = True
    if neg in dff:                       # neg measured only if it is interior
        moving_neg_flat = bool(v.sum() >= 10 and
                               negative_control_flat(np.nan_to_num(dff[neg][v], nan=0.0))["flat"])

    valid_fraction = float(v.mean())
    trace_corr = float(np.median(corrs)) if corrs else 0.0
    passed = bool(valid_fraction > 0.8 and trace_corr > 0.95 and moving_neg_flat)
    return {
        "valid_fraction": valid_fraction,
        "trace_corr_median": trace_corr,
        "moving_negative_flat": moving_neg_flat,
        "passed": passed,
    }
```

- [ ] **Step 4: Run** `pytest tests/test_calcium_v2b_validation.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): v2b acceptance battery (valid-fraction, interior trace, honest, under-constrained gating)"`

## Task 5: `stabilize_calcium_dense` tool + README

**Files:** Modify `tools/qc.py`, `README.md`; Test `tests/test_calcium_tool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_tool.py  (append)
def test_stabilize_calcium_dense_stores_table():
    POS16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
             (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
             (65, 65), (65, 100), (100, 65), (100, 100)]
    rec = make_recording(n_frames=40, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=44, motion={"lateral_px": 6.0})
    state.put_array("dmv", rec.movie)
    state.put_array("dlb", rec.labels)
    res = qc.stabilize_calcium_dense("d_tc", movie_key="dmv", labels_key="dlb")
    assert res["metrics"]["dense_table"] in state.list_tables()
    assert "valid_fraction" in res["metrics"]
    df = state.get_table(res["metrics"]["dense_table"])
    assert {"label", "time_index", "dff_corrected"} <= set(df.columns)
    assert set(df["label"]) == {13, 14, 15, 16}     # interior cells only
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# src/imajin/tools/qc.py  (append)
@tool(
    description="v2b dense motion correction: landmark-driven piecewise-affine warp "
    "(gated by density/triangle/strain/fold; bounded to the landmark hull) stabilises the "
    "movie, then measures fixed ROIs strictly inside the hull. Stores a corrected ΔF/F0 "
    "table and reports valid-frame fraction. Warp is disabled (frames gated) when "
    "under-constrained.",
    phase="6",
    worker=True,
)
def stabilize_calcium_dense(table_name: str, movie_key: str, labels_key: str) -> dict[str, Any]:
    import pandas as pd
    from imajin.analysis.calcium_motion import correct_sparse
    from imajin.analysis.calcium_warp import dense_stabilize, dense_corrected_dff

    movie = _materialize(state.get_array(movie_key))
    labels = _materialize(state.get_array(labels_key)).astype(np.int32)
    res = correct_sparse(movie, labels)
    stab = dense_stabilize(movie, labels, res)
    dff = dense_corrected_dff(stab["movie"], labels, stab["valid"])
    rows = [
        {"label": int(lbl), "time_index": t, "dff_corrected": float(v)}
        for lbl, arr in dff.items()
        for t, v in enumerate(arr)
    ]
    dense_table = state.put_table(
        f"{table_name}_dense_corrected", pd.DataFrame(rows),
        spec={"tool": "stabilize_calcium_dense", "source_table": table_name},
    )
    valid_fraction = float(np.mean(stab["valid"]))
    warnings: list[str] = []
    if valid_fraction < 0.5:
        warnings.append(f"dense warp valid on only {valid_fraction:.0%} of frames (under-constrained?)")
    if not dff:
        warnings.append("no cells strictly inside the landmark hull; nothing measured")
    metrics = {
        "kind": "calcium_dense_correction",
        "table_name": table_name,
        "dense_table": dense_table,
        "valid_fraction": valid_fraction,
        "n_interior_cells": len(dff),
        "failed": False,
    }
    return _record(table_name, warnings, metrics)
```

  README: extend the v2a calcium bullet — "and **v2b** dense piecewise-affine warp (`stabilize_calcium_dense`) for dense sheets, hull-bounded and gated by density/triangle/strain/fold checks."

- [ ] **Step 4: Run** `pytest tests/test_calcium_tool.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(calcium): stabilize_calcium_dense tool (v2b) + README"`

---

## Self-review

**Spec coverage (dense path):** density (ConvexHull-area), triangle min-angle, strain (singular values), fold (det>0) gates (Task 1 ↔ spec dense-warp gates); **no extrapolation** — warp NaN-filled outside the mesh + interior-only measurement (Tasks 2,3 ↔ "no extrapolation beyond the hull"); under-constrained → frames/ROIs gated (Tasks 2,4 ↔ "dense warp disabled → gate"); moving-negative-control-flat-after-warp + trace recovery (Task 4 ↔ reqs 2,6); tool (Task 5). Reuses v2a `correct_sparse` landmarks + `calcium_synth` affine GT.

**Placeholder scan:** every code step inline/runnable; no TBD. Reuses `_safe_corr`/`negative_control_flat` from `calcium_validation.py`.

**Type consistency:** single **(x, y)** convention throughout; `positions`/centroids (y,x) swapped before warp. `warp_quality`→`{ok,reason,n,tform}` and the SAME `tform` is applied in `dense_stabilize` (Tasks 1,2). `interior_labels`→`dict[label]->(x,y)` (Tasks 1,3). `dense_stabilize`→`{movie,valid,reason}` (Tasks 2,3,4,5). `dense_corrected_dff`→`dict[interior_label]->(T,)` (Tasks 3,4,5). Tool returns via `_record` → tests read `res["metrics"][...]` (Task 5).

## Changelog
- rev.0 (2026-06-26): initial v2b plan.
- rev.1 (2026-06-26): fixed all 5 Codex NO-GO items — single (x,y) convention with the validated tform actually applied; real density gate (ConvexHull area, ≥1/(50px)²); `mode="constant", cval=NaN` (no extrapolation) + `interior_labels` so only strictly-in-hull cells are measured; 4×4-grid test layout with interior cells (neg control interior) and late-frame pullback assertions.
- rev.2 (2026-06-26): fixed 2 residual Codex items — `dense_stabilize` builds the warp from **directly-located** landmarks only (`reason == "located"`, excluding interpolated), so density/count isn't faked; `dense_corrected_dff` requires the **full** ROI core in-mesh per frame (`np.isfinite(patch).all()`), guaranteeing the cell is inside that frame's actual (subset) hull, not just touching it.
