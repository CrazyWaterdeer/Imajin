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
