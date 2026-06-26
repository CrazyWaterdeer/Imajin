from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.calcium_validation import run_v2_acceptance

# central, non-overlapping placements that stay in-frame under the test motion
POS6 = [(45, 40), (45, 60), (45, 80), (80, 40), (80, 60), (80, 80)]


def test_v2_acceptance_recovers_and_stays_honest():
    rec = make_recording(n_frames=120, shape=(120, 120), n_cells=6, positions=POS6,
                         seed=21, bleach_tau=600.0, motion={"lateral_px": 8.0})
    rep = run_v2_acceptance(rec)
    assert rep["residual_median_px"] < 1.0            # req 1
    assert rep["trace_corr_median"] > 0.95            # req 2
    assert rep["event_amp_ratio_median"] > 0.8        # req 3 event-amplitude preserved
    assert rep["coverage_gain_pp"] > 0                # v2 beats v1 coverage on moving data
    assert rep["moving_negative_flat"] is True        # req 6a
    assert abs(rep["confidence_dynamics_corr"]) < 0.2  # req 6c
    assert rep["passed"] is True
