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
                         motion={"lateral_px": 6.0})       # < MIN_LANDMARKS cells
    rep = run_v2b_acceptance(rec)
    assert rep["valid_fraction"] < 0.2                     # warp disabled -> gated
