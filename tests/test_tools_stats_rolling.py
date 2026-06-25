import numpy as np
import pandas as pd
import pytest

from imajin import session as state
from imajin.tools import stats


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def test_rolling_f0_tracks_bleaching_and_keeps_transient():
    n = 200
    t = np.arange(n)
    f0_true = 100.0 * np.exp(-t / 600.0)          # slow bleaching baseline
    sig = f0_true.copy()
    sig[50:55] += 0.8 * f0_true[50:55]            # one transient
    df = pd.DataFrame({"label": 1, "time_index": t, "mean_intensity": sig})
    table = state.put_table("bleach", df, spec={"tool": "test"})

    res = stats.normalize_timecourse(
        table, method="delta_f_over_f0_rolling",
        f0_window=41, f0_percentile=10.0, new_table_name="bleach_dff",
    )
    out = state.get_table(res["table_name"])
    dff = out[res["output_col"]].to_numpy()
    # transient clearly separable; baseline stays small (a small positive
    # low-percentile bias is expected, so the bound is honest, not zero).
    assert dff[52] > 0.3
    assert np.nanmedian(dff[100:]) < 0.1
