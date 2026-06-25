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
