from __future__ import annotations

import pandas as pd
import pytest

from imajin.agent import state
from imajin.tools import stats


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def _endpoint_table() -> str:
    df = pd.DataFrame(
        {
            "sample_name": ["c1", "c1", "c2", "c2", "t1", "t1", "t2", "t2"],
            "group": ["control", "control", "control", "control", "treated", "treated", "treated", "treated"],
            "label": [1, 2, 1, 2, 1, 2, 1, 2],
            "mean_intensity": [1.0, 1.2, 1.1, 0.9, 3.0, 3.2, 2.8, 3.1],
        }
    )
    return state.put_table("measurements", df, spec={"tool": "test"})


def test_describe_table_creates_object_and_sample_stats() -> None:
    table = _endpoint_table()

    res = stats.describe_table(table, "mean_intensity", save_csv=False)

    assert res["n_object_rows"] == 8
    assert res["n_sample_rows"] == 4
    object_df = state.get_table(res["object_stats_table"])
    sample_df = state.get_table(res["sample_stats_table"])
    assert set(object_df["group"]) == {"control", "treated"}
    assert set(sample_df["group"]) == {"control", "treated"}
    assert "median" in object_df.columns
    assert "iqr" in object_df.columns


def test_compare_groups_defaults_to_sample_level() -> None:
    table = _endpoint_table()

    res = stats.compare_groups(
        table,
        "mean_intensity",
        n_bootstrap=200,
        save_csv=False,
    )

    assert res["data_level"] == "sample"
    assert res["group_counts"] == {"control": 2, "treated": 2}
    assert res["test"] == "welch_ttest"
    result_df = state.get_table(res["result_table"])
    row = result_df.iloc[0]
    assert row["group_a"] == "control"
    assert row["group_b"] == "treated"
    assert row["mean_difference_b_minus_a"] == pytest.approx(1.975)
    assert row["p_value"] < 0.05


def test_compare_groups_identical_constants_returns_p_one() -> None:
    df = pd.DataFrame(
        {
            "sample_name": ["c1", "c2", "t1", "t2"],
            "group": ["control", "control", "treated", "treated"],
            "mean_intensity": [5.0, 5.0, 5.0, 5.0],
        }
    )
    table = state.put_table("constant_measurements", df, spec={"tool": "test"})

    res = stats.compare_groups(table, "mean_intensity", save_csv=False)

    assert res["p_value"] == pytest.approx(1.0)
    result_df = state.get_table(res["result_table"])
    assert "identical constants" in result_df.loc[0, "warnings"]


def test_ensure_default_statistics_creates_summary_and_comparison() -> None:
    table = _endpoint_table()

    outputs = stats.ensure_default_statistics(save_csv=False)

    assert len(outputs) == 1
    assert outputs[0]["source_table"] == table
    assert outputs[0]["sample_stats_table"] in state.list_tables()
    assert outputs[0]["comparison_table"] in state.list_tables()
    compare_df = state.get_table(outputs[0]["comparison_table"])
    assert compare_df.loc[0, "p_value"] < 0.05


def test_ensure_default_statistics_adds_missing_comparison_after_summary() -> None:
    table = _endpoint_table()
    stats.describe_table(table, "mean_intensity", save_csv=False)

    outputs = stats.ensure_default_statistics(save_csv=False)

    assert len(outputs) == 1
    assert outputs[0]["object_stats_table"] is None
    assert outputs[0]["comparison_table"] in state.list_tables()


def test_normalize_timecourse_and_extract_features() -> None:
    df = pd.DataFrame(
        {
            "sample_name": ["s1"] * 6,
            "group": ["treated"] * 6,
            "label": [1, 1, 1, 2, 2, 2],
            "time_index": [0, 1, 2, 0, 1, 2],
            "mean_intensity": [10.0, 10.0, 20.0, 5.0, 5.0, 15.0],
        }
    )
    table = state.put_table("timecourse", df, spec={"tool": "test"})

    norm = stats.normalize_timecourse(
        table,
        value_col="mean_intensity",
        baseline=(0, 1),
        method="delta_f_over_f0",
    )
    norm_df = state.get_table(norm["table_name"])
    assert norm["output_col"] == "mean_intensity_delta_f_over_f0"
    assert norm_df[norm["output_col"]].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 2.0])

    features = stats.extract_timecourse_features(
        norm["table_name"],
        value_col=norm["output_col"],
        baseline_window=(0, 1),
        response_window=(1, 2),
        threshold=0.5,
        save_csv=False,
    )
    feat_df = state.get_table(features["table_name"])
    assert features["n_traces"] == 2
    label1 = feat_df[feat_df["label"] == 1].iloc[0]
    assert label1["peak_amplitude"] == pytest.approx(1.0)
    assert label1["time_to_peak"] == pytest.approx(2.0)
