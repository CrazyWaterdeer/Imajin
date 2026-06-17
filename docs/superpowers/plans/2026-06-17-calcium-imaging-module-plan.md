# Calcium Imaging Module — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the v1 calcium-imaging core — rolling-percentile ΔF/F0 plus an *honest* QC layer that gates both defocus and lateral-motion contamination, validated by a synthetic ground-truth harness — so traces are either trustworthy or explicitly rejected.

**Architecture:** New headless, pure-numpy/scipy analysis modules (`analysis/calcium_*.py`) do the algorithmic work and are unit-tested without napari/Qt. A synthetic recording generator lands first and is the acceptance gate for everything else. Thin tool wrappers in `tools/` expose the work over the existing session-table + provenance + QC-record machinery (reuse `normalize_timecourse`, `measure_intensity_over_time`, `compute_timecourse_qc`, `put_table`/`get_table`, `put_qc_record`). v2 (motion *correction*/tracking) is a later plan; v1 only *detects and gates*.

**Tech Stack:** Python 3.12, numpy, scipy, scikit-image, pandas, pytest. (No torch/Cellpose/Qt needed for v1 core or its tests.)

Follows spec `docs/superpowers/specs/2026-06-17-calcium-imaging-module-design.md` (2 Codex rounds → GO). This plan: rev.2 (2 Codex rounds; changelog at bottom).

---

## Sequencing principle

The synthetic harness (M1) lands **first** because v1 acceptance is *defined* by it (spec reqs 2,4,5 + IoU/coverage calibration). It records ground-truth gate labels (which frames are defocused, the motion) so later QC can be *scored*, not just run. v1 detects/gates only — no motion correction.

## File structure

- Create `src/imajin/analysis/calcium_synth.py` — synthetic recording generator + `SyntheticRecording` dataclass (incl. ground-truth `defocus_frames`, `motion`). Pure numpy/scipy.
- Create `src/imajin/analysis/calcium_qc.py` — focus metrics + composite, per-frame cell-locate, lateral-validity, combined per-(cell,frame) gate with coverage + missingness pattern + longest contiguous run.
- Create `src/imajin/analysis/calcium_events.py` — detrended event detector, negative-control-flat, event-preservation.
- Create `src/imajin/analysis/calcium_validation.py` — v1 acceptance battery scoring gate accuracy, F0 bias, artifact, event-preservation, coverage.
- Modify `src/imajin/session.py` — add `put_array`/`get_array` (small array handoff store, mirrors `put_table`).
- Modify `src/imajin/tools/stats.py` — rolling-percentile F0 in `normalize_timecourse`.
- Modify `src/imajin/tools/qc.py` — `assess_calcium_timecourse` tool.
- Modify `src/imajin/tools/figures.py` — ΔF/F0 raster/heatmap plot.
- Create `docs/calcium_manual_reference_labels.md`; modify `README.md`.
- Tests: `tests/test_calcium_synth.py`, `tests/test_calcium_qc.py`, `tests/test_calcium_events.py`, `tests/test_calcium_validation.py`, plus additions to `tests/test_tools_stats.py`, `tests/test_tools_qc.py`, `tests/test_tools_figures.py`, `tests/test_session.py`.

### Shared signatures (authoritative; consistent across tasks)

```python
# calcium_synth.py
@dataclass
class SyntheticRecording:
    movie: np.ndarray                    # (T, Y, X) float32
    labels: np.ndarray                   # (Y, X) int   (fixed ROI footprints)
    true_dff: dict[int, np.ndarray]      # label -> (T,) true ΔF/F0
    event_frames: dict[int, list[int]]   # label -> sorted peak frame indices
    f0: dict[int, float]                 # label -> true baseline intensity
    negative_label: int | None
    defocus_frames: list[int]            # GROUND TRUTH: frames that were blurred
    motion: dict | None                  # GROUND TRUTH: motion params (or None)
    meta: dict

def make_recording(*, n_frames=200, shape=(128,128), n_cells=6, seed=0,
                   bleach_tau=None, noise=2.0, motion=None, defocus=None,
                   negative_control=True) -> SyntheticRecording: ...

# calcium_qc.py
def focus_metrics(patch) -> dict[str, float]: ...               # lap_norm, tenengrad, snr
def composite_focus(metrics_over_time: dict[str, np.ndarray]) -> np.ndarray: ...  # (T,) z-score composite
def locate_cell(frame, template, roi_centroid, search_radius=6) -> dict: ...      # dy, dx, peak, centroid  (NO footprint)
def lateral_valid(footprint, roi_mask, centroid, roi_centroid, roi_radius,
                  iou_thresh=0.7, drift_frac=0.5) -> dict: ...   # iou, drift, ok, reason
def gate_traces(movie, labels, *, search_radius=6, iou_thresh=0.7,
                drift_frac=0.5, focus_z=3.0) -> "GateResult": ...

@dataclass
class GateResult:
    usable: dict[int, np.ndarray]        # label -> (T,) bool
    coverage: dict[int, float]           # label -> fraction usable
    reason: dict[int, np.ndarray]        # label -> (T,) object[str]
    longest_run: dict[int, int]          # label -> longest contiguous usable run (frames)
    missing_frac: dict[int, float]       # label -> fraction gated

# calcium_events.py
def detect_events(trace, noise_sigma=None, k=3.0, min_len=2) -> list[tuple[int,int]]: ...  # detrended
def negative_control_flat(trace, artifact_ceiling=0.05) -> dict: ...  # flat, n_events, max_abs (detrended)
def event_preservation_rate(usable, event_frames) -> float: ...
```

---

## Milestone 1 — synthetic harness (the acceptance gate)

### Task 1: Synthetic recording — cells, transients, noise, bleaching, ground-truth labels

**Files:**
- Create: `src/imajin/analysis/calcium_synth.py`
- Test: `tests/test_calcium_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_synth.py
import numpy as np
from imajin.analysis.calcium_synth import make_recording, SyntheticRecording


def test_basic_recording_shapes_events_and_truth():
    rec = make_recording(n_frames=120, shape=(64, 64), n_cells=4, seed=1)
    assert isinstance(rec, SyntheticRecording)
    assert rec.movie.shape == (120, 64, 64)
    assert rec.labels.shape == (64, 64)
    assert set(np.unique(rec.labels)) - {0} == set(rec.true_dff)
    assert rec.negative_label in rec.true_dff
    assert np.allclose(rec.true_dff[rec.negative_label], 0.0)
    assert rec.defocus_frames == []                    # ground-truth labels exist
    assert rec.motion is None
    assert any(len(v) > 0 for k, v in rec.event_frames.items()
               if k != rec.negative_label)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_synth.py -q`
Expected: FAIL — `ModuleNotFoundError: imajin.analysis.calcium_synth`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/analysis/calcium_synth.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticRecording:
    movie: np.ndarray
    labels: np.ndarray
    true_dff: dict[int, np.ndarray]
    event_frames: dict[int, list[int]]
    f0: dict[int, float]
    negative_label: int | None
    defocus_frames: list[int] = field(default_factory=list)
    motion: dict | None = None
    meta: dict = field(default_factory=dict)


def _disk(cy, cx, radius, shape):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def _transient(n_frames, peaks, amp, tau):
    t = np.arange(n_frames, dtype=np.float64)
    trace = np.zeros(n_frames)
    for p in peaks:
        trace += amp * np.exp(-(t - p) / tau) * (t >= p)
    return trace


def make_recording(*, n_frames=200, shape=(128, 128), n_cells=6, seed=0,
                   bleach_tau=None, noise=2.0, motion=None, defocus=None,
                   negative_control=True) -> SyntheticRecording:
    rng = np.random.default_rng(seed)
    Y, X = shape
    labels = np.zeros(shape, dtype=np.int32)
    f0, true_dff, event_frames = {}, {}, {}
    radius = 5.0
    margin = int(radius) + 3
    base = np.full((n_frames, Y, X), 5.0, dtype=np.float64)

    negative_label = n_cells if negative_control else None
    for lbl in range(1, n_cells + 1):
        cy = int(rng.integers(margin, Y - margin))
        cx = int(rng.integers(margin, X - margin))
        mask = _disk(cy, cx, radius, shape)
        labels[mask] = lbl
        f0[lbl] = float(rng.uniform(40.0, 80.0))
        if negative_label is not None and lbl == negative_label:
            dff, peaks = np.zeros(n_frames), []
        else:
            n_ev = int(rng.integers(2, 5))
            peaks = sorted(int(p) for p in rng.integers(5, n_frames - 5, size=n_ev))
            dff = _transient(n_frames, peaks, amp=rng.uniform(0.4, 1.2), tau=8.0)
        true_dff[lbl] = dff
        event_frames[lbl] = peaks
        intensity = f0[lbl] * (1.0 + dff)
        base[:, mask] += intensity[:, None]

    if bleach_tau:
        base *= np.exp(-np.arange(n_frames) / float(bleach_tau))[:, None, None]

    from scipy.ndimage import gaussian_filter, shift as nd_shift
    movie = base.copy()
    if motion:
        amp = float(motion.get("lateral_px", 0.0))
        for t in range(n_frames):
            frac = t / max(1, n_frames - 1)
            movie[t] = nd_shift(movie[t], (amp * frac, amp * frac * 0.5),
                                order=1, mode="nearest")
    defocus_frames = list(defocus.get("frames", [])) if defocus else []
    if defocus_frames:
        sigma = float(defocus.get("sigma", 3.0))
        for t in defocus_frames:
            movie[t] = gaussian_filter(movie[t], sigma=sigma)

    movie = (movie + rng.normal(0.0, float(noise), size=movie.shape)).astype(np.float32)
    return SyntheticRecording(
        movie=movie, labels=labels, true_dff=true_dff, event_frames=event_frames,
        f0=f0, negative_label=negative_label, defocus_frames=defocus_frames,
        motion=motion, meta={"seed": seed, "noise": noise, "bleach_tau": bleach_tau},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_synth.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_synth.py tests/test_calcium_synth.py
git commit -m "feat(calcium): synthetic recording generator with ground-truth gate labels"
```

### Task 2: Verify motion + defocus ground truth

**Files:**
- Test: `tests/test_calcium_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_synth.py  (append)
def test_motion_and_defocus_recorded_and_applied():
    still = make_recording(n_frames=30, shape=(64, 64), n_cells=3, seed=2)
    moved = make_recording(n_frames=30, shape=(64, 64), n_cells=3, seed=2,
                          motion={"lateral_px": 4.0},
                          defocus={"frames": [15], "sigma": 4.0})
    assert moved.motion == {"lateral_px": 4.0}
    assert moved.defocus_frames == [15]
    assert not np.allclose(still.movie[-1], moved.movie[-1])
    from scipy.ndimage import sobel
    assert np.abs(sobel(moved.movie[15])).mean() < np.abs(sobel(moved.movie[14])).mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_synth.py::test_motion_and_defocus_recorded_and_applied -q`
Expected: PASS already if Task 1 implemented motion/defocus; if a stripped Task 1 was committed, FAIL → implement the motion/defocus block shown in Task 1 Step 3.

- [ ] **Step 3: (only if failing) ensure the motion/defocus block from Task 1 Step 3 is present.**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_synth.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** (skip if Task 1 already covered it)

```bash
git add tests/test_calcium_synth.py
git commit -m "test(calcium): assert synthetic motion/defocus ground truth"
```

---

## Milestone 2 — rolling-percentile F0

### Task 3: Rolling-percentile F0 in `normalize_timecourse`

**Files:**
- Modify: `src/imajin/tools/stats.py:782` (`normalize_timecourse`)
- Test: `tests/test_tools_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_stats.py  (append)
import numpy as np


def test_rolling_f0_tracks_bleaching_and_keeps_transient():
    n = 200
    t = np.arange(n)
    f0_true = 100.0 * np.exp(-t / 600.0)            # slow bleaching
    sig = f0_true.copy()
    sig[50:55] += 0.8 * f0_true[50:55]              # one transient
    df = pd.DataFrame({"label": 1, "time_index": t, "mean_intensity": sig})
    table = state.put_table("bleach", df, spec={"tool": "test"})

    res = stats.normalize_timecourse(table, method="delta_f_over_f0_rolling",
                                     f0_window=41, f0_percentile=10.0,
                                     new_table_name="bleach_dff")
    out = state.get_table(res["table_name"])
    dff = out[res["output_col"]].to_numpy()
    # transient clearly separable; baseline stays small (a small positive
    # low-percentile bias is expected, so the bound is honest, not zero).
    assert dff[52] > 0.3
    assert np.nanmedian(dff[100:]) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_stats.py::test_rolling_f0_tracks_bleaching_and_keeps_transient -q`
Expected: FAIL — `delta_f_over_f0_rolling` not accepted.

- [ ] **Step 3: Write minimal implementation**

In `normalize_timecourse`: (a) extend the `method` Literal with `"f_over_f0_rolling"`, `"delta_f_over_f0_rolling"`; (b) add params `f0_window: int | None = None`, `f0_percentile: float = 10.0`; (c) add the two keys to the `out_col` default-name dict (`f"{value_col}_f_over_f0_rolling"`, `f"{value_col}_delta_f_over_f0_rolling"`); (d) guard the fixed-baseline computation: `if baseline is None and method not in rolling_methods:` (define `rolling_methods` first); for rolling methods set `baseline_times = set()` and skip it. (e) **At the very top of the per-group loop body, before `base_mask`/`base_vals` are computed, handle rolling and `continue`:**

```python
    rolling_methods = {"f_over_f0_rolling", "delta_f_over_f0_rolling"}

    def _rolling_f0(vals, window, pct):
        n = len(vals)
        w = max(3, int(window) | 1)
        half = w // 2
        out_f0 = np.empty(n, dtype=float)
        for i in range(n):
            seg = vals[max(0, i - half): min(n, i + half + 1)]
            seg = seg[np.isfinite(seg)]
            out_f0[i] = np.nanpercentile(seg, pct) if seg.size else np.nan
        return out_f0
```

Inside `for _key, group in out.groupby(...)`, immediately after `vals = ...` is built:

```python
        if method in rolling_methods:
            window = f0_window or max(3, (int(round(len(vals) * 0.1)) | 1))
            f0_series = _rolling_f0(vals, window, f0_percentile)
            out.loc[idx, f"{out_col}_baseline"] = f0_series
            safe = np.where(f0_series != 0, f0_series, np.nan)
            norm = vals / safe if method == "f_over_f0_rolling" else (vals - safe) / safe
            out.loc[idx, out_col] = norm
            continue
```

(The existing `base_mask`/`base_vals`/`bad_baseline` block now only runs for the non-rolling methods, so no trace is skipped before the rolling branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_stats.py -q`
Expected: PASS (new test + existing stats tests green).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/stats.py tests/test_tools_stats.py
git commit -m "feat(calcium): rolling-percentile F0 option in normalize_timecourse"
```

---

## Milestone 3 — honest QC gates

### Task 4: Focus metrics + composite

**Files:**
- Create: `src/imajin/analysis/calcium_qc.py`
- Test: `tests/test_calcium_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_qc.py
import numpy as np
from scipy.ndimage import gaussian_filter
from imajin.analysis.calcium_qc import focus_metrics, composite_focus


def test_focus_metrics_drop_when_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.normal(50, 10, size=(24, 24)).astype(np.float32)
    blurred = gaussian_filter(sharp, sigma=3.0)
    assert focus_metrics(sharp)["tenengrad"] > focus_metrics(blurred)["tenengrad"]
    assert focus_metrics(sharp)["lap_norm"] > focus_metrics(blurred)["lap_norm"]


def test_composite_focus_is_low_on_blurred_frame():
    rng = np.random.default_rng(1)
    frames = [focus_metrics(rng.normal(50, 10, size=(24, 24))) for _ in range(10)]
    frames[5] = focus_metrics(gaussian_filter(rng.normal(50, 10, size=(24, 24)), 3.0))
    series = {k: np.array([f[k] for f in frames]) for k in frames[0]}
    comp = composite_focus(series)
    assert comp[5] == comp.min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_qc.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/analysis/calcium_qc.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import laplace, shift as nd_shift, sobel


def focus_metrics(patch) -> dict[str, float]:
    p = np.asarray(patch, dtype=np.float64)
    pn = p / (float(np.mean(p)) or 1.0)        # intensity-normalized → activity-robust
    lap = laplace(pn)
    gy, gx = sobel(pn, axis=0), sobel(pn, axis=1)
    std = float(np.std(p)) or 1.0
    snr = (float(np.percentile(p, 90)) - float(np.percentile(p, 10))) / std
    return {"lap_norm": float(np.var(lap)),
            "tenengrad": float(np.mean(gy ** 2 + gx ** 2)),
            "snr": float(snr)}


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x)) or 1.0
    return (x - float(np.mean(x))) / sd


def composite_focus(metrics_over_time: dict[str, np.ndarray]) -> np.ndarray:
    zs = [_zscore(v) for v in metrics_over_time.values()]
    return np.mean(np.vstack(zs), axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_qc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_qc.py tests/test_calcium_qc.py
git commit -m "feat(calcium): intensity-normalized focus metrics + multi-metric composite"
```

### Task 5: Per-frame cell-locate (cross-correlation)

**Files:**
- Modify: `src/imajin/analysis/calcium_qc.py`
- Test: `tests/test_calcium_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_qc.py  (append)
from scipy.ndimage import shift as nd_shift
from imajin.analysis.calcium_qc import locate_cell


def test_locate_recovers_known_shift():
    rng = np.random.default_rng(3)
    template = rng.normal(0, 1, size=(15, 15))
    frame = np.zeros((40, 40)); frame[12:27, 12:27] = template
    shifted = nd_shift(frame, (3, -2), order=1, mode="nearest")
    res = locate_cell(shifted, template, roi_centroid=(19.0, 19.0), search_radius=6)
    assert round(res["dy"]) == 3 and round(res["dx"]) == -2
    assert res["peak"] > 0.5
    assert set(res) == {"dy", "dx", "peak", "centroid"}      # no phantom footprint key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_qc.py::test_locate_recovers_known_shift -q`
Expected: FAIL — `locate_cell` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# append to calcium_qc.py  (note: nd_shift/sobel/laplace already imported in Task 4)
def locate_cell(frame, template, roi_centroid, search_radius=6) -> dict:
    frame = np.asarray(frame, dtype=np.float64)
    tmpl = np.asarray(template, dtype=np.float64); tmpl = tmpl - tmpl.mean()
    th, tw = tmpl.shape
    cy, cx = int(round(roi_centroid[0])), int(round(roi_centroid[1]))
    best = (-np.inf, 0, 0)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            y0, x0 = cy + dy - th // 2, cx + dx - tw // 2
            if y0 < 0 or x0 < 0 or y0 + th > frame.shape[0] or x0 + tw > frame.shape[1]:
                continue
            win = frame[y0:y0 + th, x0:x0 + tw]; win = win - win.mean()
            denom = (np.linalg.norm(win) * np.linalg.norm(tmpl)) or 1.0
            score = float(np.sum(win * tmpl) / denom)
            if score > best[0]:
                best = (score, dy, dx)
    peak, dy, dx = best
    return {"dy": dy, "dx": dx, "peak": peak,
            "centroid": (roi_centroid[0] + dy, roi_centroid[1] + dx)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_qc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_qc.py tests/test_calcium_qc.py
git commit -m "feat(calcium): normalized cross-correlation cell-locate"
```

### Task 6: Lateral-validity gate (drift = `drift_frac × roi_radius`)

**Files:**
- Modify: `src/imajin/analysis/calcium_qc.py`
- Test: `tests/test_calcium_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_qc.py  (append)
from imajin.analysis.calcium_qc import lateral_valid


def test_lateral_valid_flags_large_drift():
    roi = np.zeros((40, 40), bool); roi[15:25, 15:25] = True
    foot_ok = np.zeros((40, 40), bool); foot_ok[16:25, 16:25] = True
    ok = lateral_valid(foot_ok, roi, (20.0, 20.0), (20.0, 20.0), roi_radius=5.0)
    assert ok["ok"] and ok["iou"] >= 0.7
    foot_off = np.zeros((40, 40), bool); foot_off[25:34, 25:34] = True
    bad = lateral_valid(foot_off, roi, (29.0, 29.0), (20.0, 20.0), roi_radius=5.0)
    assert not bad["ok"]
    assert not lateral_valid(None, roi, (20.0, 20.0), (20.0, 20.0), 5.0)["ok"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_qc.py::test_lateral_valid_flags_large_drift -q`
Expected: FAIL — `lateral_valid` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# append to calcium_qc.py
def lateral_valid(footprint, roi_mask, centroid, roi_centroid, roi_radius,
                  iou_thresh=0.7, drift_frac=0.5) -> dict:
    if footprint is None:
        return {"iou": 0.0, "drift": np.inf, "ok": False, "reason": "unlocatable"}
    fp, roi = np.asarray(footprint, bool), np.asarray(roi_mask, bool)
    inter = float(np.count_nonzero(fp & roi))
    union = float(np.count_nonzero(fp | roi)) or 1.0
    iou = inter / union
    drift = float(np.hypot(centroid[0] - roi_centroid[0], centroid[1] - roi_centroid[1]))
    ok = (iou >= iou_thresh) and (drift < drift_frac * roi_radius)   # spec: 0.5 × radius
    reason = "ok" if ok else ("low_iou" if iou < iou_thresh else "drift")
    return {"iou": iou, "drift": drift, "ok": ok, "reason": reason}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_qc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_qc.py tests/test_calcium_qc.py
git commit -m "feat(calcium): lateral-validity IoU/drift gate (drift = 0.5 x radius)"
```

### Task 7: Combined gate + coverage + missingness pattern

**Files:**
- Modify: `src/imajin/analysis/calcium_qc.py`
- Test: `tests/test_calcium_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_qc.py  (append)
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_qc import gate_traces, GateResult


def test_gate_high_coverage_when_still():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5)
    res = gate_traces(rec.movie, rec.labels)
    assert isinstance(res, GateResult)
    for lbl in rec.true_dff:
        assert res.coverage[lbl] > 0.9
        assert res.longest_run[lbl] >= 50


def test_gate_drops_defocused_window():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5,
                        defocus={"frames": [24, 25, 26], "sigma": 5.0})  # minority outliers
    res = gate_traces(rec.movie, rec.labels)
    assert sum(not res.usable[lbl][25] for lbl in rec.true_dff) >= 3


def test_gate_drops_laterally_moved_cell():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5,
                        motion={"lateral_px": 12.0})    # large slide off the ROI
    res = gate_traces(rec.movie, rec.labels)
    # late frames (after big drift) should be gated for most cells
    assert np.mean([res.usable[lbl][-1] for lbl in rec.true_dff]) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_qc.py -q -k gate`
Expected: FAIL — `gate_traces`/`GateResult` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# append to calcium_qc.py
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


def _longest_true_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def gate_traces(movie, labels, *, search_radius=6, iou_thresh=0.7,
                drift_frac=0.5, focus_z=3.0) -> GateResult:
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
        # multi-metric composite focus over time (req 2)
        metrics = {k: np.empty(T) for k in ("lap_norm", "tenengrad", "snr")}
        for t in range(T):
            fm = focus_metrics(movie[t, y0:y1, x0:x1])
            for k in metrics:
                metrics[k][t] = fm[k]
        comp = composite_focus(metrics)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_qc.py -q`
Expected: PASS (all qc tests incl. defocus + lateral acceptance).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_qc.py tests/test_calcium_qc.py
git commit -m "feat(calcium): combined multi-metric gate + coverage + longest-run"
```

---

## Milestone 4 — events + acceptance battery

### Task 8: Detrended event detector, negative-control-flat, event-preservation

**Files:**
- Create: `src/imajin/analysis/calcium_events.py`
- Test: `tests/test_calcium_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_events.py
import numpy as np
from imajin.analysis.calcium_events import (
    detect_events, negative_control_flat, event_preservation_rate)


def test_detect_and_flat_with_offset():
    trace = np.full(200, 0.03)            # small positive DC offset (low-pct F0 bias)
    trace[50:56] += 0.8                   # a real transient on top
    ev = detect_events(trace, k=4.0)
    assert any(s <= 52 <= e for s, e in ev)
    # a flat-but-offset noisy control must read flat after detrending
    flat = negative_control_flat(0.03 + np.random.default_rng(0).normal(0, 0.01, 200))
    assert flat["flat"]
    assert not negative_control_flat(trace)["flat"]


def test_event_preservation():
    usable = np.ones(100, bool); usable[40:60] = False
    rate = event_preservation_rate({1: usable}, {1: [10, 50, 80]})
    assert rate == 2 / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_events.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/analysis/calcium_events.py
from __future__ import annotations

import numpy as np


def _detrend(trace):
    x = np.asarray(trace, dtype=float)
    return x - float(np.median(x))            # remove DC offset (low-pct F0 bias)


def detect_events(trace, noise_sigma=None, k=3.0, min_len=2) -> list[tuple[int, int]]:
    x = _detrend(trace)
    if noise_sigma is None:
        d = np.diff(x)
        noise_sigma = float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2)) or 1e-9
    above = x > k * noise_sigma
    events, start = [], None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                events.append((start, i - 1))
            start = None
    if start is not None and len(above) - start >= min_len:
        events.append((start, len(above) - 1))
    return events


def negative_control_flat(trace, artifact_ceiling=0.05) -> dict:
    x = _detrend(trace)
    ev = detect_events(trace, k=4.0)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    return {"flat": bool(len(ev) == 0 and max_abs < artifact_ceiling),
            "n_events": len(ev), "max_abs": max_abs}


def event_preservation_rate(usable, event_frames) -> float:
    total = kept = 0
    for lbl, frames in event_frames.items():
        mask = usable.get(lbl)
        for f in frames:
            total += 1
            if mask is not None and 0 <= f < len(mask) and mask[f]:
                kept += 1
    return (kept / total) if total else 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_events.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_events.py tests/test_calcium_events.py
git commit -m "feat(calcium): detrended event detector, neg-control-flat, event-preservation"
```

### Task 9: v1 acceptance battery (scores gate accuracy, F0, artifact, coverage)

**Files:**
- Create: `src/imajin/analysis/calcium_validation.py`
- Test: `tests/test_calcium_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calcium_validation.py
from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_validation import run_v1_acceptance


def test_v1_acceptance_scores_and_passes():
    rec = make_recording(n_frames=150, shape=(96, 96), n_cells=6, seed=7,
                        bleach_tau=600.0, noise=2.0,
                        defocus={"frames": [60, 61, 62], "sigma": 4.0})
    rep = run_v1_acceptance(rec)
    assert rep["negative_control_flat"] is True          # req 8 hard gate
    assert rep["event_preservation"] >= 0.95             # req 2 binding
    assert rep["defocus_recall"] >= 0.9                  # req 2 gating accuracy
    assert 0.0 <= rep["f0_bias_negative"] < 0.1          # req 5
    assert rep["artifact_max"] < 0.05                    # req 4 reported + gated
    assert rep["passed"] is True
    assert set(rep["coverage"]) == set(rec.true_dff)     # req 3 reported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calcium_validation.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/analysis/calcium_validation.py
from __future__ import annotations

import numpy as np

from imajin.analysis.calcium_qc import gate_traces
from imajin.analysis.calcium_events import negative_control_flat, event_preservation_rate


def _rolling_dff(intensity, window=41, pct=10.0):
    n = len(intensity); half = window // 2
    f0 = np.array([np.percentile(intensity[max(0, i - half): i + half + 1], pct)
                   for i in range(n)])
    return (intensity - f0) / np.where(f0 != 0, f0, np.nan)


def run_v1_acceptance(rec) -> dict:
    gate = gate_traces(rec.movie, rec.labels)
    neg = rec.negative_label

    # defocus gating accuracy vs ground truth (union over cells)
    T = rec.movie.shape[0]
    truth_def = np.zeros(T, bool); truth_def[rec.defocus_frames] = True
    pred_def = np.zeros(T, bool)
    for lbl, reason in gate.reason.items():
        pred_def |= (reason == "defocus")
    tp = int(np.sum(pred_def & truth_def))
    recall = tp / max(1, int(truth_def.sum()))
    precision = tp / max(1, int(pred_def.sum())) if pred_def.any() else 1.0

    # negative control: flat + F0 bias + artifact magnitude on usable frames
    neg_flat, f0_bias, artifact_max = True, 0.0, 0.0
    if neg is not None:
        roi = rec.labels == neg
        inten = rec.movie[:, roi].mean(axis=1)
        dff = _rolling_dff(inten)
        usable = gate.usable[neg]
        trace = np.nan_to_num(dff[usable], nan=0.0)
        nc = negative_control_flat(trace)
        neg_flat = nc["flat"]
        artifact_max = float(nc["max_abs"])
        f0_bias = float(abs(np.nanmedian(dff[usable])))

    signalling = {k: v for k, v in rec.event_frames.items() if k != neg}
    preservation = event_preservation_rate(gate.usable, signalling)

    passed = bool(neg_flat and preservation >= 0.95 and recall >= 0.9
                  and f0_bias < 0.1 and artifact_max < 0.05)
    return {
        "negative_control_flat": bool(neg_flat),
        "event_preservation": float(preservation),
        "defocus_recall": float(recall),
        "defocus_precision": float(precision),
        "f0_bias_negative": float(f0_bias),
        "artifact_max": float(artifact_max),
        "coverage": {int(k): float(v) for k, v in gate.coverage.items()},
        "longest_run": {int(k): int(v) for k, v in gate.longest_run.items()},
        "passed": passed,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calcium_validation.py -q`
Expected: PASS. (If `defocus_recall` < 0.9, raise the synthetic defocus `sigma` or lower `focus_z`; if it clips a true event lower `event_preservation`, move the defocus window off event peaks — tune the *synthetic*, never the binding threshold.)

- [ ] **Step 5: Commit**

```bash
git add src/imajin/analysis/calcium_validation.py tests/test_calcium_validation.py
git commit -m "feat(calcium): v1 acceptance battery (gate accuracy, F0 bias, artifact, coverage)"
```

---

## Milestone 5 — session array store, tool wiring, viz, docs

### Task 10a: `put_array`/`get_array` on the session

**Files:**
- Modify: `src/imajin/session.py` (next to `put_table`, ~line 645)
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py  (append; this file already exercises session state)
import numpy as np
from imajin import session as state


def test_put_get_array_roundtrip():
    state.put_array("m", np.zeros((3, 4)))
    assert state.get_array("m").shape == (3, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py::test_put_get_array_roundtrip -q`
Expected: FAIL — `put_array` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/session.py  (add near put_table; uses the existing current_session())
def put_array(key: str, arr) -> str:
    current_session().arrays[key] = arr
    return key


def get_array(key: str):
    return current_session().arrays[key]
```

Add an `arrays: dict[str, Any] = field(default_factory=dict)` field to the `AnalysisSession` dataclass (mirror the existing `tables` field). If a custom `__init__`/reset path exists, initialise `arrays` there too.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/session.py tests/test_session.py
git commit -m "feat(session): put_array/get_array for headless array handoff"
```

### Task 10b: `assess_calcium_timecourse` tool

**Files:**
- Modify: `src/imajin/tools/qc.py`
- Test: `tests/test_tools_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_qc.py  (append; uses this file's autouse reset fixture)
import numpy as np, pandas as pd
from imajin.tools import qc
from imajin.analysis.calcium_synth import make_recording


def test_assess_calcium_timecourse_reports_coverage():
    rec = make_recording(n_frames=40, shape=(64, 64), n_cells=3, seed=9)
    state.put_array("ca_movie", rec.movie)
    state.put_array("ca_labels", rec.labels)
    df = pd.DataFrame({"label": [1], "time_index": [0], "mean_intensity": [1.0]})
    table = state.put_table("ca_tc", df, spec={"tool": "test"})

    res = qc.assess_calcium_timecourse(table, movie_key="ca_movie", labels_key="ca_labels")
    assert len(res["metrics"]["coverage"]) == 3          # nested under metrics (via _record)
    assert "longest_run" in res["metrics"]
    assert res["status"] in {"pass", "warning", "fail"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_qc.py::test_assess_calcium_timecourse_reports_coverage -q`
Expected: FAIL — `assess_calcium_timecourse` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/imajin/tools/qc.py  (append)
@tool(
    description="Assess a calcium timecourse: gate defocus and lateral-motion frames, "
    "report per-cell usable coverage, longest contiguous run, and missing fraction; "
    "store a QC record. Detection only; does not correct motion.",
    phase="6",
    worker=True,
)
def assess_calcium_timecourse(table_name: str, movie_key: str, labels_key: str,
                              coverage_floor: float = 0.5,
                              min_run: int = 10) -> dict[str, Any]:
    from imajin.analysis.calcium_qc import gate_traces

    movie = _materialize(state.get_array(movie_key))
    labels = _materialize(state.get_array(labels_key)).astype(np.int32)
    gate = gate_traces(movie, labels)
    coverage = {int(k): float(v) for k, v in gate.coverage.items()}
    longest = {int(k): int(v) for k, v in gate.longest_run.items()}
    missing = {int(k): float(v) for k, v in gate.missing_frac.items()}
    rejected = [k for k in coverage
                if coverage[k] < coverage_floor or longest[k] < min_run]
    warnings: list[str] = []
    if rejected:
        warnings.append(f"{len(rejected)} cell(s) rejected (coverage<{coverage_floor} "
                        f"or longest run<{min_run}): {rejected}")
    metrics = {"kind": "calcium_timecourse", "table_name": table_name,
               "coverage": coverage, "longest_run": longest, "missing_frac": missing,
               "rejected": rejected, "failed": False}
    return _record(table_name, warnings, metrics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_qc.py -q`
Expected: PASS (new test + existing qc tests).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/qc.py tests/test_tools_qc.py
git commit -m "feat(calcium): assess_calcium_timecourse tool (coverage + longest-run gating)"
```

### Task 11: ΔF/F0 heatmap plot

**Files:**
- Modify: `src/imajin/tools/figures.py`
- Test: `tests/test_tools_figures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_figures.py  (append; match this file's existing import style)
import numpy as np, pandas as pd
from imajin.tools import figures


def test_dff_heatmap_writes_png(tmp_path):
    rows = [{"label": lbl, "time_index": t,
             "mean_intensity_delta_f_over_f0": np.sin(t / 3 + lbl)}
            for lbl in (1, 2, 3) for t in range(20)]
    table = state.put_table("dfftc", pd.DataFrame(rows), spec={"tool": "test"})
    res = figures.plot_dff_heatmap(table, value_col="mean_intensity_delta_f_over_f0",
                                   out_path=str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").exists() and res["n_traces"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_figures.py::test_dff_heatmap_writes_png -q`
Expected: FAIL — `plot_dff_heatmap` missing.

- [ ] **Step 3: Write minimal implementation**

First check `figures.py`'s existing imports: it imports the session helpers directly (e.g. `from imajin.session import get_table`). Use that **same reference** (`get_table`, not `state.get_table`):

```python
# src/imajin/tools/figures.py  (append; reuse the file's existing Agg/matplotlib pattern)
@tool(description="ΔF/F0 raster/heatmap: one row per ROI, time on x, colour = ΔF/F0.",
      phase="6", worker=True)
def plot_dff_heatmap(table_name: str, value_col: str, out_path: str | None = None,
                     time_col: str | None = None) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from imajin.tools._dataframes import infer_time_column

    df = get_table(table_name)                      # NB: figures.py imports get_table directly
    tcol = time_col or infer_time_column(df)
    piv = df.pivot_table(index="label", columns=tcol, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(piv.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xlabel(tcol); ax.set_ylabel("ROI"); fig.colorbar(im, ax=ax, label=value_col)
    path = out_path or f"{table_name}_dff_heatmap.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return {"path": path, "n_traces": int(piv.shape[0])}
```

If `figures.py` actually imports `from imajin import session as state`, use `state.get_table` instead — match whatever that file already does (verify in Step 3 before writing).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_figures.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/figures.py tests/test_tools_figures.py
git commit -m "feat(calcium): ΔF/F0 raster/heatmap plot"
```

### Task 12: Manual-reference label spec + README note

**Files:**
- Create: `docs/calcium_manual_reference_labels.md`
- Modify: `README.md`

- [ ] **Step 1: Write the doc** — `docs/calcium_manual_reference_labels.md` specifying the labelled-data deliverable needed to compute reqs 2/6/7 on *real* data: (a) per-frame defocus labels on ≥2 hard gut clips — CSV `clip,frame,cell,defocus`; (b) hand-tracked cell identities across frames — CSV `clip,frame,cell,y,x`; (c) hand-drawn ROI label TIFFs for ≥10 cells. Location: `tests/data/calcium_ref/`. State that until these exist, reqs 2/6/7 are validated on synthetic only.

- [ ] **Step 2: Update README** — add to Features: "Calcium imaging (v1): rolling-percentile ΔF/F0; honest defocus + lateral-motion gating with per-cell coverage, longest-run, and missing-fraction reporting; synthetic validation harness."

- [ ] **Step 3: Commit**

```bash
git add docs/calcium_manual_reference_labels.md README.md
git commit -m "docs(calcium): manual-reference label spec + README v1 note"
```

---

## Self-review

**Spec coverage:** rolling F0 (Task 3 ↔ F0/req 5); multi-metric defocus gate (Tasks 4,7 ↔ req 2); lateral gate incl. footprint=shifted-ROI + unlocatable→gate (Tasks 5,6,7 ↔ req 1 + v1-footprint decision); coverage **+ missingness pattern (missing_frac) + longest contiguous run** reported and used for reject (Tasks 7,10b ↔ req 3, the round-1 regression now fixed); synthetic harness w/ ground-truth labels (Tasks 1,2 ↔ Validation §1); negative-control hard gate + event-preservation + defocus recall/precision + F0 bias + artifact (Tasks 8,9 ↔ reqs 8,2,4,5); array handoff (Task 10a); tool wiring (Task 10b); ΔF/F0 viz (Task 11); manual-reference labels (Task 12 ↔ Validation §4). **Deferred to v2 (correctly out of scope):** motion *correction*/warp, IDF1 tracking (reqs 6,7), btrack landmark linking, stimulus-window-conditional coverage reject (needs stimulus metadata — pattern is still *reported* in v1).

**Placeholder scan:** every code step has runnable code; no TBD/TODO. Task 12 is a prose deliverable.

**Type consistency:** `locate_cell` returns exactly `{dy,dx,peak,centroid}` (no `footprint`) — asserted in Task 5, consumed in Task 7 where the footprint is built by shifting the ROI mask. `GateResult(usable,coverage,reason,longest_run,missing_frac)` identical in Tasks 7,9,10b. `lateral_valid` keys `{iou,drift,ok,reason}` consistent Tasks 6,7. `composite_focus(dict)->ndarray` consistent Tasks 4,7. `event_preservation_rate(usable, event_frames)` consistent Tasks 8,9. rolling method name `delta_f_over_f0_rolling` consistent Task 3 ↔ Task 9 `_rolling_dff` mirror. Tool return nests metrics via `_record` → tests assert `res["metrics"][...]` (Task 10b).

## Known v1 simplifications (carried to v2 plan, not regressions)
- Footprint = temporal-min-template-driven shifted ROI; Cellpose-cadence re-seg fallback deferred to v2.
- IoU 0.7 / focus_z 3.0 / min_run 10 are starting defaults; Task 9's harness is where they get calibrated against trace-error impact before any real-data claim.
- Stimulus-window-conditional coverage rejection deferred (needs stimulus metadata); the missingness *pattern* (missing_frac, longest_run) is reported now.

## Changelog
- rev.0 (2026-06-17): initial v1 plan from approved spec.
- rev.1 (2026-06-17): fixed all 12 Codex NO-GO items — rolling-F0 branch ordering + honest test bound; drift threshold `0.5×radius`; footprint=shifted ROI mask + `locate_cell` signature corrected (no phantom `footprint`); multi-metric composite focus gate; lateral-gating end-to-end test; detrended events/neg-control; expanded acceptance battery (defocus recall/precision, F0 bias, artifact); explicit `session.put_array/get_array`; tool return/test shape reconciled; `figures` uses `get_table`; coverage missingness-pattern + longest-run restored to v1 (req 3).
- rev.2 (2026-06-17): fixed 3 residual Codex items — added `nd_shift` import to the locate test; robust median/MAD defocus threshold + minority-defocus test (plain z-score bottomed above −3 when 10/60 frames blurred); acceptance battery now reports & gates `artifact_max` (not just a boolean). Documented limitation: the focus gate assumes defocus is a minority of frames; majority-defocus shows up as low coverage instead.
