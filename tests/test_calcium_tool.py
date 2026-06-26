import pandas as pd
import pytest

from imajin import session as state
from imajin.tools import qc
from imajin.analysis.calcium_synth import make_recording


@pytest.fixture(autouse=True)
def _clean():
    state.reset_tables()
    yield
    state.reset_tables()


def test_assess_calcium_timecourse_reports_coverage():
    rec = make_recording(n_frames=40, shape=(64, 64), n_cells=3, seed=9)
    state.put_array("ca_movie", rec.movie)
    state.put_array("ca_labels", rec.labels)
    df = pd.DataFrame({"label": [1], "time_index": [0], "mean_intensity": [1.0]})
    table = state.put_table("ca_tc", df, spec={"tool": "test"})

    res = qc.assess_calcium_timecourse(table, movie_key="ca_movie", labels_key="ca_labels")
    assert len(res["metrics"]["coverage"]) == 3
    assert "longest_run" in res["metrics"]
    assert res["status"] in {"pass", "warning", "fail"}


def test_correct_calcium_motion_stores_corrected_table():
    rec = make_recording(n_frames=40, shape=(90, 90), n_cells=4, seed=14,
                         motion={"lateral_px": 8.0})
    state.put_array("mv", rec.movie)
    state.put_array("lb", rec.labels)
    res = qc.correct_calcium_motion("mv_tc", movie_key="mv", labels_key="lb")
    assert len(res["metrics"]["coverage"]) == 4
    assert res["metrics"]["corrected_table"] in state.list_tables()
    df = state.get_table(res["metrics"]["corrected_table"])
    assert {"label", "time_index", "dff_corrected"} <= set(df.columns)


def test_stabilize_calcium_dense_stores_table():
    pos16 = [(30, 30), (30, 65), (30, 100), (30, 135), (65, 30), (65, 135),
             (100, 30), (100, 135), (135, 30), (135, 65), (135, 100), (135, 135),
             (65, 65), (65, 100), (100, 65), (100, 100)]
    rec = make_recording(n_frames=40, shape=(160, 160), n_cells=16, positions=pos16,
                         seed=44, motion={"lateral_px": 6.0})
    state.put_array("dmv", rec.movie)
    state.put_array("dlb", rec.labels)
    res = qc.stabilize_calcium_dense("d_tc", movie_key="dmv", labels_key="dlb")
    assert res["metrics"]["dense_table"] in state.list_tables()
    assert "valid_fraction" in res["metrics"]
    df = state.get_table(res["metrics"]["dense_table"])
    assert {"label", "time_index", "dff_corrected"} <= set(df.columns)
    assert set(df["label"]) == {13, 14, 15, 16}      # interior cells only
