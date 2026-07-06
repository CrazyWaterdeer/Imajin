from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from imajin import session as state
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


def test_describe_table_writes_long_format(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin import session as state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools import stats

    reset_process_bundle()
    bundle = start_analysis(name="stattest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
                "max_intensity": [3.0, 3.1, 4.5, 4.6],
            }
        ),
        spec={"tool": "test"},
    )

    stats.describe_table("measurements", "mean_intensity")
    stats.describe_table("measurements", "max_intensity")

    df = pd.read_csv(bundle / "stats" / "describe__measurements.csv")
    assert {"mean_intensity", "max_intensity"} <= set(df["value_col"])
    # Object-level rows for both groups, both value_cols → 4 rows minimum.
    object_rows = df[df["level"] == "object"]
    assert len(object_rows) >= 4
    # No flat per-value_col stats files.
    assert not any(p.name.startswith("stats_object__") for p in (bundle / "stats").iterdir())
    reset_process_bundle()


def test_compare_groups_writes_long_format(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin import session as state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools import stats

    reset_process_bundle()
    bundle = start_analysis(name="cmptest")
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

    stats.compare_groups("measurements", "mean_intensity")
    df = pd.read_csv(bundle / "stats" / "compare__measurements.csv")
    assert df.iloc[0]["value_col"] == "mean_intensity"
    assert df.iloc[0]["p_value"] < 0.05
    reset_process_bundle()


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


# --- paired mode (inside/outside within-sample) --------------------------------------

def _paired_inside_outside_table(n: int = 6, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = float(rng.uniform(1.0, 3.0))
        rows.append({"sample_name": f"s{i}", "region": "inside", "val": base + float(rng.uniform(0.3, 0.9))})
        rows.append({"sample_name": f"s{i}", "region": "outside", "val": base})
    return state.put_table("io", pd.DataFrame(rows), spec={"tool": "test"})


def test_paired_wilcoxon_matches_scipy() -> None:
    df = state.get_table(_paired_inside_outside_table())
    res = stats.compare_groups("io", "val", group_col="region", test="wilcoxon", n_bootstrap=200)

    piv = df.pivot(index="sample_name", columns="region", values="val")
    inside = piv["inside"].to_numpy(float)
    outside = piv["outside"].to_numpy(float)
    W, p = scipy_stats.wilcoxon(outside, inside)  # group_a=inside (sorted), group_b=outside -> (b,a)

    row = state.get_table(res["result_table"]).iloc[0]
    assert row["test"] == "wilcoxon_signed_rank"
    assert row["group_a"] == "inside" and row["group_b"] == "outside"
    assert row["n_pairs"] == 6
    assert res["p_value"] == pytest.approx(float(p))
    assert row["statistic"] == pytest.approx(float(W))
    assert res["group_counts"] == {"inside": 6, "outside": 6}


def test_paired_t_matches_scipy_and_reference_group_sets_direction() -> None:
    df = state.get_table(_paired_inside_outside_table())
    # reference_group=outside -> group_a=outside, group_b=inside, d = inside - outside > 0
    res = stats.compare_groups(
        "io", "val", group_col="region", test="paired_t",
        reference_group="outside", n_bootstrap=200,
    )
    piv = df.pivot(index="sample_name", columns="region", values="val")
    inside = piv["inside"].to_numpy(float)
    outside = piv["outside"].to_numpy(float)
    t, p = scipy_stats.ttest_rel(inside, outside)  # (b=inside, a=outside)

    row = state.get_table(res["result_table"]).iloc[0]
    assert row["group_a"] == "outside" and row["group_b"] == "inside"
    assert row["mean_difference_b_minus_a"] > 0  # inside enriched
    assert res["p_value"] == pytest.approx(float(p))
    assert row["statistic"] == pytest.approx(float(t))
    assert row["cohens_dz"] > 0


def test_signed_rank_biserial_sign_and_value() -> None:
    # all differences positive -> rank-biserial = +1
    d = np.array([0.5, 0.8, 0.2, 1.1, 0.6])
    assert stats._signed_rank_biserial(d) == pytest.approx(1.0)
    # mixed: exact Rp/Rn from |d| ranks
    d2 = np.array([2.0, -1.0, 3.0])  # ranks of |d|: 1->2, -1->1, 3->3 ; Rp=2+3=5, Rn=1
    assert stats._signed_rank_biserial(d2) == pytest.approx((5 - 1) / (5 + 1))
    assert stats._signed_rank_biserial(np.zeros(4)) == 0.0


def test_paired_bootstrap_ci_deterministic_and_brackets_mean() -> None:
    lo, hi = stats._bootstrap_mean_paired(np.array([0.5, 0.7, 0.6, 0.8, 0.9]), n_bootstrap=500, seed=7)
    lo2, hi2 = stats._bootstrap_mean_paired(np.array([0.5, 0.7, 0.6, 0.8, 0.9]), n_bootstrap=500, seed=7)
    assert (lo, hi) == (lo2, hi2)  # deterministic with fixed seed
    assert lo <= 0.7 <= hi  # brackets the mean difference (~0.7)


def test_paired_drops_incomplete_pair_with_warning() -> None:
    df = pd.DataFrame(
        {
            "sample_name": ["s1", "s1", "s2", "s2", "s3"],  # s3 has only inside
            "region": ["inside", "outside", "inside", "outside", "inside"],
            "val": [2.0, 1.0, 2.5, 1.2, 3.0],
        }
    )
    state.put_table("io", df, spec={"tool": "test"})
    res = stats.compare_groups("io", "val", group_col="region", test="paired_t", n_bootstrap=100)
    row = state.get_table(res["result_table"]).iloc[0]
    assert row["n_pairs"] == 2
    assert row["n_dropped_incomplete"] == 1
    assert any("incomplete" in w for w in res["warnings"])


def test_paired_auto_stays_independent() -> None:
    _paired_inside_outside_table()
    res = stats.compare_groups("io", "val", group_col="region", n_bootstrap=100)  # test=auto
    assert res["test"] == "welch_ttest"


def test_paired_object_level_rejects_duplicates() -> None:
    # forcing object level on a multi-object-per-sample table -> unpairable duplicates
    df = pd.DataFrame(
        {
            "sample_name": ["s1", "s1", "s1", "s1"],
            "region": ["inside", "inside", "outside", "outside"],
            "val": [2.0, 2.2, 1.0, 1.1],
        }
    )
    state.put_table("io", df, spec={"tool": "test"})
    with pytest.raises(ValueError):
        stats.compare_groups("io", "val", group_col="region", test="wilcoxon", level="object")


def test_paired_raw_object_duplicates_ok_at_sample_level() -> None:
    # many cells per sample x region is fine at auto/sample level (aggregated first)
    df = pd.DataFrame(
        {
            "sample_name": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
            "region": ["inside", "inside", "outside", "outside"] * 2,
            "val": [2.0, 2.2, 1.0, 1.1, 2.5, 2.7, 1.3, 1.2],
        }
    )
    state.put_table("io", df, spec={"tool": "test"})
    res = stats.compare_groups("io", "val", group_col="region", test="paired_t", n_bootstrap=100)
    assert state.get_table(res["result_table"]).iloc[0]["n_pairs"] == 2


def test_paired_requires_two_pairs() -> None:
    df = pd.DataFrame(
        {"sample_name": ["s1", "s1"], "region": ["inside", "outside"], "val": [2.0, 1.0]}
    )
    state.put_table("io", df, spec={"tool": "test"})
    with pytest.raises(ValueError):
        stats.compare_groups("io", "val", group_col="region", test="wilcoxon")


def test_paired_missing_sample_col_errors() -> None:
    df = pd.DataFrame({"region": ["inside", "outside", "inside", "outside"], "val": [2.0, 1.0, 2.5, 1.2]})
    state.put_table("io", df, spec={"tool": "test"})
    with pytest.raises(ValueError):
        stats.compare_groups("io", "val", group_col="region", sample_col="sample_name", test="wilcoxon")


def test_pseudoreplication_warning_on_clustered_object_level() -> None:
    df = pd.DataFrame(
        {
            "sample_name": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "region": ["inside", "inside", "outside", "inside", "outside", "outside"],
            "val": [2.0, 2.1, 1.0, 2.5, 1.1, 1.2],
        }
    )
    state.put_table("io", df, spec={"tool": "test"})
    res = stats.compare_groups("io", "val", group_col="region", level="object", n_bootstrap=100)
    assert any("pseudoreplication" in w for w in res["warnings"])
