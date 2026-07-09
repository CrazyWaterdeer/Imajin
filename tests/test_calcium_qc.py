import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_qc import (
    focus_metrics, composite_focus, locate_cell, lateral_valid,
    gate_traces, GateResult)


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


def test_locate_recovers_known_shift():
    rng = np.random.default_rng(3)
    template = rng.normal(0, 1, size=(15, 15))
    frame = np.zeros((40, 40)); frame[12:27, 12:27] = template
    shifted = nd_shift(frame, (3, -2), order=1, mode="nearest")
    res = locate_cell(shifted, template, roi_centroid=(19.0, 19.0), search_radius=6)
    assert round(res["dy"]) == 3 and round(res["dx"]) == -2
    assert res["peak"] > 0.5
    assert set(res) == {"dy", "dx", "peak", "centroid"}


def test_lateral_valid_flags_large_drift():
    roi = np.zeros((40, 40), bool); roi[15:25, 15:25] = True
    foot_ok = np.zeros((40, 40), bool); foot_ok[16:25, 16:25] = True
    ok = lateral_valid(foot_ok, roi, (20.0, 20.0), (20.0, 20.0), roi_radius=5.0)
    assert ok["ok"] and ok["iou"] >= 0.7
    foot_off = np.zeros((40, 40), bool); foot_off[25:34, 25:34] = True
    bad = lateral_valid(foot_off, roi, (29.0, 29.0), (20.0, 20.0), roi_radius=5.0)
    assert not bad["ok"]
    assert not lateral_valid(None, roi, (20.0, 20.0), (20.0, 20.0), 5.0)["ok"]


def test_gate_high_coverage_when_still():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5)
    res = gate_traces(rec.movie, rec.labels)
    assert isinstance(res, GateResult)
    for lbl in rec.true_dff:
        assert res.coverage[lbl] > 0.9
        assert res.longest_run[lbl] >= 50


def test_gate_drops_defocused_window():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5,
                         defocus={"frames": [24, 25, 26], "sigma": 5.0})
    res = gate_traces(rec.movie, rec.labels)
    assert sum(not res.usable[lbl][25] for lbl in rec.true_dff) >= 3


def test_gate_drops_laterally_moved_cell():
    rec = make_recording(n_frames=60, shape=(80, 80), n_cells=4, seed=5,
                         motion={"lateral_px": 12.0})
    res = gate_traces(rec.movie, rec.labels)
    assert np.mean([res.usable[lbl][-1] for lbl in rec.true_dff]) < 0.5


def test_locate_cell_numba_matches_reference():
    # The numba NCC locate kernel must make the SAME (dy, dx) decision as the numpy
    # reference across many frames/templates/centroids; peak matches to fp rounding.
    from imajin.analysis.calcium_qc import locate_cell, _locate_cell_reference

    rng = np.random.default_rng(0)
    H, W, th = 120, 120, 11
    max_peak_diff = 0.0
    for _ in range(300):
        frame = rng.normal(5.0, 1.0, (H, W))
        cy, cx = int(rng.integers(20, H - 20)), int(rng.integers(20, W - 20))
        yy, xx = np.mgrid[0:H, 0:W]
        sy, sx = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
        frame += 40.0 * np.exp(-(((yy - cy - sy) ** 2 + (xx - cx - sx) ** 2) / (2 * 3.0 ** 2)))
        tmpl = frame[cy - th // 2:cy + th // 2 + 1, cx - th // 2:cx + th // 2 + 1].copy()
        c = (float(cy), float(cx))
        a = locate_cell(frame, tmpl, c, search_radius=6)
        b = _locate_cell_reference(frame, tmpl, c, search_radius=6)
        assert (a["dy"], a["dx"]) == (b["dy"], b["dx"])
        max_peak_diff = max(max_peak_diff, abs(a["peak"] - b["peak"]))
    assert max_peak_diff < 1e-9
