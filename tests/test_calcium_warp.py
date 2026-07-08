import numpy as np
from scipy.ndimage import center_of_mass

from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_motion import correct_sparse
from imajin.analysis.calcium_warp import (
    warp_quality, interior_labels, dense_stabilize, dense_corrected_dff)

POS16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
         (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
         (65, 65), (65, 100), (100, 65), (100, 100)]


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
    yy, xx = np.mgrid[0:160, 0:160]
    for i, (cy, cx) in enumerate(POS16, start=1):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25] = i
    inner = interior_labels(labels, margin=4.0)
    assert set(inner) == {13, 14, 15, 16}        # only the inner 2x2 are strictly in-hull


def test_dense_stabilize_pulls_interior_cell_back_on_a_moving_frame():
    rec = make_recording(n_frames=30, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=31, motion={"lateral_px": 6.0}, noise=0.0)
    res = correct_sparse(rec.movie, rec.labels)
    stab = dense_stabilize(rec.movie, rec.labels, res)
    assert stab["valid"].mean() > 0.8
    t = int(np.where(stab["valid"])[0][-1])          # a LATE valid frame (motion present)
    base = np.array(center_of_mass(rec.labels == 13))   # interior cell, (row, col)
    f0 = float(rec.f0[13])
    frame = np.nan_to_num(stab["movie"][t], nan=0.0)
    yy, xx = np.mgrid[0:160, 0:160]
    win = (np.abs(yy - base[0]) < 7) & (np.abs(xx - base[1]) < 7) & (frame > 0.5 * f0)
    com = np.array(center_of_mass(frame * win))
    assert np.hypot(*(com - base)) < 2.0


def test_dense_corrected_dff_recovers_interior_trace():
    rec = make_recording(n_frames=90, shape=(160, 160), n_cells=16, positions=POS16,
                         seed=32, motion={"lateral_px": 6.0})
    res = correct_sparse(rec.movie, rec.labels)
    stab = dense_stabilize(rec.movie, rec.labels, res)
    out = dense_corrected_dff(stab["movie"], rec.labels, stab["valid"])
    assert set(out) == {13, 14, 15, 16}              # only interior cells measured
    lbl = max((k for k in (13, 14, 15) if rec.event_frames[k]),
              key=lambda k: len(rec.event_frames[k]))
    v = stab["valid"]
    r = np.corrcoef(np.nan_to_num(out[lbl][v]), rec.true_dff[lbl][v])[0, 1]
    assert r > 0.95


def test_rolling_percentile_numba_matches_numpy_reference():
    # The numba F0 kernel must match the numpy reference (the previous inline
    # loop) to floating-point rounding, including NaN gating and edge truncation.
    from imajin.analysis.calcium_warp import (
        _rolling_percentile,
        _rolling_percentile_numpy,
    )

    rng = np.random.default_rng(0)
    inten = rng.normal(100.0, 10.0, size=600)
    inten[rng.random(600) < 0.2] = np.nan   # scattered gated frames
    inten[:15] = np.nan                     # leading gap (edge truncation)
    inten[-8:] = np.nan                     # trailing gap
    for window, pct in ((41, 10.0), (21, 5.0), (7, 50.0), (1, 90.0)):
        fast = _rolling_percentile(inten, window, pct)
        ref = _rolling_percentile_numpy(inten, window, pct)
        np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_masked_mean_numba_matches_numpy_reference():
    # The numba per-frame ROI-mean kernel must match the numpy reference (the
    # previous inline movie[t][m] loop), including NaN gating and invalid frames.
    from imajin.analysis.calcium_warp import (
        _masked_mean_over_time,
        _masked_mean_over_time_numpy,
    )

    rng = np.random.default_rng(1)
    movie = rng.normal(50.0, 5.0, size=(120, 40, 40))
    movie[rng.random(movie.shape) < 0.02] = np.nan   # scattered out-of-bounds pixels
    valid = rng.random(120) > 0.15
    yy, xx = np.mgrid[0:40, 0:40]
    m = (yy - 20) ** 2 + (xx - 18) ** 2 <= 5.0 ** 2
    rows, cols = np.nonzero(m)

    fast = _masked_mean_over_time(movie, rows, cols, valid)
    ref = _masked_mean_over_time_numpy(movie, rows, cols, valid)
    np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-9, equal_nan=True)
