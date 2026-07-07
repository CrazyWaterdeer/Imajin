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


def _debris_table() -> str:
    # one large main region + many small debris per sample. Main regions are equal
    # across groups; virgin just has MORE debris objects. Unweighted per-sample means
    # deflate more for virgin (spurious difference); area weighting recovers equality.
    rows: list[dict] = []
    for s, main in [("m1", 8000.0), ("m2", 7800.0), ("m3", 8200.0)]:
        rows.append({"sample_name": s, "group": "mated", "area": 1000.0, "mean_intensity": main})
        rows += [{"sample_name": s, "group": "mated", "area": 5.0, "mean_intensity": 4000.0}] * 5
    for s, main in [("v1", 8000.0), ("v2", 7900.0), ("v3", 8100.0)]:
        rows.append({"sample_name": s, "group": "virgin", "area": 1000.0, "mean_intensity": main})
        rows += [{"sample_name": s, "group": "virgin", "area": 5.0, "mean_intensity": 4000.0}] * 15
    return state.put_table("debris", pd.DataFrame(rows), spec={"tool": "test"})


def test_compare_groups_auto_weights_by_area() -> None:
    table = _debris_table()
    res = stats.compare_groups(table, "mean_intensity", group_col="group", save_csv=False)

    assert res["weighted_by"] == "area"  # auto-detected the regionprops area column
    assert res["data_level"] == "sample"
    # main regions are equal across groups; area weighting recovers that -> not significant
    assert res["p_value"] > 0.2
    assert any("area-weighted" in w for w in res["warnings"])


def test_compare_groups_weight_none_is_unweighted_and_more_significant() -> None:
    table = _debris_table()
    weighted = stats.compare_groups(table, "mean_intensity", group_col="group", save_csv=False)
    unweighted = stats.compare_groups(
        table, "mean_intensity", group_col="group", weight_col=None, save_csv=False
    )

    assert unweighted["weighted_by"] is None
    # the debris artifact: unweighted comparison is far more "significant" than weighted
    assert unweighted["p_value"] < weighted["p_value"]


def test_compare_groups_auto_is_unweighted_without_area_column() -> None:
    table = _endpoint_table()  # has no `area` column
    res = stats.compare_groups(table, "mean_intensity", save_csv=False)
    assert res["weighted_by"] is None


def test_resolve_weight_col() -> None:
    df = pd.DataFrame({"area": [1.0], "mean_intensity": [2.0], "v": [3.0]})
    assert stats.resolve_weight_col(df, "auto", "mean_intensity") == "area"
    # auto never weights the value by itself
    assert stats.resolve_weight_col(df, "auto", "area") is None
    assert stats.resolve_weight_col(df.drop(columns="area"), "auto", "v") is None
    assert stats.resolve_weight_col(df, None, "v") is None
    assert stats.resolve_weight_col(df, "v", "mean_intensity") == "v"
    with pytest.raises(ValueError, match="weight_col"):
        stats.resolve_weight_col(df, "missing", "v")


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


# --- assumption- and n-aware auto test selection -------------------------------------

def _groups_table(data: dict[str, list[float]], name: str = "g") -> str:
    rows = []
    for grp, vals in data.items():
        for i, v in enumerate(vals):
            rows.append({"sample_name": f"{grp}{i}", "group": grp, "val": float(v)})
    return state.put_table(name, pd.DataFrame(rows), spec={"tool": "test"})


def test_auto_welch_for_normal_two_groups() -> None:
    _groups_table({"a": [8, 9, 10, 11, 12, 10], "b": [18, 19, 20, 21, 22, 20]})
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "welch_ttest"
    sel = res["test_selection"]
    assert sel["auto_selected_test"] == "welch"
    assert sel["normality_shapiro_p"]["a"] > 0.05


def test_auto_mannwhitney_for_nonnormal_two_groups() -> None:
    _groups_table({"a": [1, 1, 1, 1, 1, 50], "b": [2, 2, 2, 2, 2, 60]})
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "mann_whitney_u"
    assert any("normality" in w.lower() for w in res["warnings"])


def test_auto_anova_for_normal_three_groups() -> None:
    # consistency fix: 3 normal groups get parametric ANOVA (old auto always used Kruskal)
    _groups_table({"a": [8, 9, 10, 11, 12], "b": [18, 19, 20, 21, 22], "c": [28, 29, 30, 31, 32]})
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "one_way_anova"


def test_auto_kruskal_for_nonnormal_three_groups() -> None:
    _groups_table({"a": [1, 1, 1, 1, 50], "b": [2, 3, 4, 5, 6], "c": [3, 4, 5, 6, 7]})
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "kruskal_wallis"


def test_auto_small_n_warns_and_stays_parametric() -> None:
    _groups_table({"a": [10, 11], "b": [20, 21]})  # n=2 per group -> normality unverifiable
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "welch_ttest"  # parametric default when normality can't be assessed
    assert res["test_selection"]["n_min"] == 2
    assert any("n=2" in w or "power" in w.lower() for w in res["warnings"])


def test_within_subject_design_warns_on_independent_test() -> None:
    rows = []
    for i in range(4):  # s0..s3 measured in BOTH groups -> paired structure
        rows += [
            {"sample_name": f"s{i}", "group": "inside", "val": 5.0 + i},
            {"sample_name": f"s{i}", "group": "outside", "val": 3.0 + i},
        ]
    state.put_table("ws", pd.DataFrame(rows), spec={"tool": "test"})
    res = stats.compare_groups("ws", "val", group_col="group", save_csv=False)
    assert any("paired" in w.lower() and "within-subject" in w.lower() for w in res["warnings"])


# --- post-hoc pairwise (3+ groups) ---------------------------------------------------

def test_posthoc_games_howell_for_anova() -> None:
    rng = np.random.default_rng(1)
    _groups_table({
        "A": list(10 + rng.normal(0, 1, 8)),
        "B": list(20 + rng.normal(0, 1, 8)),
        "C": list(10.5 + rng.normal(0, 1, 8)),  # close to A
    })
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "one_way_anova"
    assert all(r["method"] == "games_howell" for r in res["posthoc"])
    ph = {(r["group_a"], r["group_b"]): r["p_adjusted"] for r in res["posthoc"]}
    assert ph[("A", "C")] > 0.05  # A and C don't differ
    assert ph[("A", "B")] < 0.05 and ph[("B", "C")] < 0.05  # both differ from B
    assert res["posthoc_table"] is not None
    assert len(state.get_table(res["posthoc_table"])) == 3  # 3 pairs


def test_posthoc_dunn_holm_for_kruskal() -> None:
    _groups_table({
        "A": [1, 1, 1, 1, 1, 1, 1, 50],
        "B": [9, 9, 9, 9, 9, 9, 9, 90],
        "C": [1, 1, 1, 2, 1, 1, 1, 55],
    })
    res = stats.compare_groups("g", "val", save_csv=False)
    assert res["test"] == "kruskal_wallis"
    assert all(r["method"] == "dunn+holm" for r in res["posthoc"])
    # Holm adjustment never lowers a p-value below its raw value
    assert all(r["p_adjusted"] >= r["p_value"] - 1e-12 for r in res["posthoc"])


def test_posthoc_can_be_disabled_and_absent_for_two_groups() -> None:
    _groups_table({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    off = stats.compare_groups("g", "val", posthoc=False, save_csv=False)
    assert off["posthoc"] is None and off["posthoc_table"] is None

    _groups_table({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, name="two")
    two = stats.compare_groups("two", "val", save_csv=False)
    assert two["posthoc"] is None  # no post-hoc for a single pair


def test_adjust_pvalues_holm_and_bh() -> None:
    holm = stats._adjust_pvalues([0.01, 0.02, 0.03], "holm")
    assert holm == [pytest.approx(0.03), pytest.approx(0.04), pytest.approx(0.04)]
    assert holm == sorted(holm)  # step-down: monotone non-decreasing
    bh = stats._adjust_pvalues([0.01, 0.02, 0.03], "fdr_bh")
    assert all(v == pytest.approx(0.03) for v in bh)
    assert stats._adjust_pvalues([], "holm") == []
    assert stats._adjust_pvalues([0.5], "none") == [0.5]


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


def test_paired_auto_stays_independent_but_warns() -> None:
    _paired_inside_outside_table()
    res = stats.compare_groups("io", "val", group_col="region", n_bootstrap=100)  # test=auto
    # auto never silently switches to a paired test (that is the user's design assertion)...
    assert res["test"] in {"welch_ttest", "mann_whitney_u"}
    assert "wilcoxon" not in res["test"] and "paired" not in res["test"]
    # ...but it now flags the within-subject design so the user can pick a paired test
    assert any("paired" in w.lower() for w in res["warnings"])


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
