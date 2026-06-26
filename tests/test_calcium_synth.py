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
    assert rec.defocus_frames == []
    assert rec.motion is None
    assert any(len(v) > 0 for k, v in rec.event_frames.items()
               if k != rec.negative_label)


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


def test_true_positions_match_pixels_under_affine():
    from scipy.ndimage import center_of_mass
    pos_in = [(30, 30), (30, 60), (60, 30), (60, 60)]   # central → stay in-frame under motion
    rec = make_recording(n_frames=30, shape=(90, 90), n_cells=4, positions=pos_in,
                         seed=4, motion={"lateral_px": 8.0, "shear": 0.05}, noise=0.0)
    for lbl in rec.true_dff:
        pos = rec.true_positions[lbl]
        assert pos.shape == (30, 2)
        cy, cx = pos[-1]
        f0 = float(rec.f0[lbl])
        last = rec.movie[-1]
        yy, xx = np.mgrid[0:90, 0:90]
        win = (np.abs(yy - cy) < 8) & (np.abs(xx - cx) < 8) & (last > 0.5 * f0)
        assert win.sum() > 5
        com = center_of_mass(last * win)
        assert np.hypot(com[0] - cy, com[1] - cx) < 1.5


def test_silent_window_makes_cell_disappear():
    rec = make_recording(n_frames=40, shape=(64, 64), n_cells=3, seed=6, noise=0.0,
                         silent_windows={1: (10, 20)})
    assert np.allclose(rec.true_dff[1][10:20], 0.0)
    roi = rec.labels == 1
    # during the window the cell vanishes toward background (not ~f0) -> unobservable
    assert rec.movie[15][roi].mean() < 0.5 * rec.f0[1]
