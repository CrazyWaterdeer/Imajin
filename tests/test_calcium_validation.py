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
