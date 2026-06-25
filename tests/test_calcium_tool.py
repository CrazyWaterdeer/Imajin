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
