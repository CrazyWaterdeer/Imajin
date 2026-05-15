from __future__ import annotations

import pandas as pd
import pytest

from imajin.agent import state
from imajin.tools import figures


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


@pytest.fixture(autouse=True)
def _require_matplotlib():
    pytest.importorskip("matplotlib")


def test_plot_group_distribution_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "distribution.svg"

    res = figures.plot_group_distribution(
        table,
        "mean_intensity",
        output_path=str(out),
    )

    assert res["path"] == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert res["data_level"] == "sample"
    assert res["plot_data_table"] in state.list_tables()
    assert res["p_value"] < 0.05
    assert res["stats_result_table"] in state.list_tables()


def test_plot_timecourse_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "timecourse",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c1", "c1", "t1", "t1", "t1"],
                "group": ["control", "control", "control", "treated", "treated", "treated"],
                "label": [1, 1, 1, 1, 1, 1],
                "time_index": [0, 1, 2, 0, 1, 2],
                "delta_f": [0.0, 0.1, 0.2, 0.0, 0.5, 1.0],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "timecourse.svg"

    res = figures.plot_timecourse(
        table,
        value_col="delta_f",
        output_path=str(out),
    )

    assert res["path"] == str(out)
    assert out.exists()
    assert res["unit_level"] == "sample"
    assert res["plot_data_table"] in state.list_tables()


def test_plot_scatter_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "ch1": [1.0, 2.0, 3.0, 4.0],
                "ch2": [2.0, 4.0, 6.0, 8.0],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "scatter.svg"

    res = figures.plot_scatter(table, "ch1", "ch2", output_path=str(out))

    assert res["path"] == str(out)
    assert out.exists()
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["pearson_p_value"] == pytest.approx(0.0)
    assert res["fit_slope"] == pytest.approx(2.0)


def test_figure_writes_into_active_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from pathlib import Path
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.results import read_bundle_metadata
    from imajin.tools import figures

    reset_process_bundle()
    bundle = start_analysis(name="figtest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )

    res = figures.plot_group_distribution("measurements", "mean_intensity")

    out = Path(res["path"])
    assert out.parent == bundle / "figures"
    assert out.exists()
    outputs = read_bundle_metadata(bundle)["outputs"]
    assert any(o["kind"] == "figure" and o["path"] == f"figures/{out.name}" for o in outputs)

    reset_process_bundle()
