import numpy as np

from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_motion import (
    observability, motion_safe_template, propagated_locate, neighbour_interpolate,
    correct_sparse, corrected_dff, CorrectionResult)

# central, well-separated placements that stay in-frame under the test motions
POS5 = [(40, 40), (40, 75), (75, 40), (75, 75), (57, 57)]


def test_observability_flags_low_contrast():
    rng = np.random.default_rng(0)
    bg = 2.0
    bright = np.full((11, 11), 5.0)
    bright[3:8, 3:8] = 60.0
    bright += rng.normal(0, bg, bright.shape)
    dim = np.full((11, 11), 5.0) + rng.normal(0, bg, (11, 11))
    assert observability(bright, bg)["observable"]
    assert not observability(dim, bg)["observable"]


def test_propagated_locate_follows_large_drift():
    rec = make_recording(n_frames=40, shape=(110, 110), n_cells=3,
                         positions=[(45, 45), (45, 75), (75, 45)], seed=11,
                         motion={"lateral_px": 16.0})
    lbl = 1
    roi = rec.labels == lbl
    tmpl = motion_safe_template(rec.movie, roi)
    res = propagated_locate(rec.movie, roi, tmpl)
    assert set(res) == {"centroid", "peak"}
    err = np.hypot(*(res["centroid"] - rec.true_positions[lbl]).T)
    assert np.median(err) < 1.5


def test_neighbour_interpolate_valid_and_invalid():
    t0 = np.array([10.0, 10.0])
    n0 = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], float)
    nt = n0 + np.array([5.0, -3.0])
    ok = neighbour_interpolate(t0, n0, nt)
    assert ok["ok"] and np.allclose(ok["xy"], t0 + [5.0, -3.0], atol=0.5)
    one_sided = neighbour_interpolate(
        t0, np.array([[0, 0], [2, 0], [0, 2]], float),
        np.array([[0, 0], [2, 0], [0, 2]], float) + 5)
    assert not one_sided["ok"]


def test_correct_sparse_tracks_moving_visible_cells():
    rec = make_recording(n_frames=50, shape=(110, 110), n_cells=5, positions=POS5,
                         seed=12, motion={"lateral_px": 10.0})
    res = correct_sparse(rec.movie, rec.labels)
    assert isinstance(res, CorrectionResult)
    for lbl in rec.true_dff:
        u = res.usable[lbl]
        assert u.mean() > 0.8
        err = np.hypot(*(res.positions[lbl] - rec.true_positions[lbl]).T)
        assert np.median(err[u]) < 1.5


def test_correct_sparse_interpolates_silent_moving_cell():
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
    rec = make_recording(n_frames=40, shape=(90, 90), n_cells=2, seed=17,
                         motion={"lateral_px": 8.0}, silent_windows={1: (15, 30)})
    res = correct_sparse(rec.movie, rec.labels)
    w = slice(15, 30)
    assert res.usable[1][w].mean() < 0.3
    assert not any(str(r) == "interpolated" for r in res.reason[1][w])


def test_corrected_dff_recovers_moving_trace():
    rec = make_recording(n_frames=90, shape=(110, 110), n_cells=5, positions=POS5,
                         seed=13, motion={"lateral_px": 10.0})
    res = correct_sparse(rec.movie, rec.labels)
    out = corrected_dff(rec.movie, rec.labels, res)
    lbl = max((k for k in rec.event_frames if k != rec.negative_label),
              key=lambda k: len(rec.event_frames[k]))
    u = res.usable[lbl]
    r = np.corrcoef(np.nan_to_num(out[lbl][u]), rec.true_dff[lbl][u])[0, 1]
    assert r > 0.9
