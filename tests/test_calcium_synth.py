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
