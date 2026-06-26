from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imajin import session as state
from imajin.tools import figures


@pytest.fixture(autouse=True)
def _clean():
    state.reset_tables()
    yield
    state.reset_tables()


@pytest.fixture(autouse=True)
def _require_matplotlib():
    pytest.importorskip("matplotlib")


def test_dff_heatmap_writes_png(tmp_path):
    rows = [
        {"label": lbl, "time_index": t,
         "mean_intensity_delta_f_over_f0": float(np.sin(t / 3 + lbl))}
        for lbl in (1, 2, 3)
        for t in range(20)
    ]
    table = state.put_table("dfftc", pd.DataFrame(rows), spec={"tool": "test"})
    out = tmp_path / "h.png"

    res = figures.plot_dff_heatmap(
        table, value_col="mean_intensity_delta_f_over_f0", output_path=str(out),
    )

    assert res["path"] == str(out)
    assert out.exists()
    assert res["n_traces"] == 3
