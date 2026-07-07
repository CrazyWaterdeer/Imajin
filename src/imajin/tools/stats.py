from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from imajin.session import get_table, get_table_entry, list_tables, put_table
from imajin.result_bundles import register_stats_rows
from imajin.results import slugify_result_name
from imajin.tools._dataframes import finite_numeric_frame, infer_time_column
from imajin.tools.registry import tool


SummaryLevel = Literal["auto", "sample", "object"]
SampleAgg = Literal["mean", "median"]

_STAT_TOOL_NAMES = {
    "describe_table",
    "compare_groups",
    "summarize_experiment",
}

_NON_MEASUREMENT_TABLE_TOOLS = {
    *_STAT_TOOL_NAMES,
    "batch_auto_statistics_input",
    "plot_group_distribution",
    "plot_timecourse",
    "plot_scatter",
    "filter_table",
}

_NON_MEASUREMENT_TABLE_OPS = {
    "skeleton_nodes",
    "skeleton_edges",
    "skeleton_components",
    "extract_branch_metrics",
    "compute_sholl_analysis",
    "track_cells",
}

_EXCLUDED_VALUE_COLUMNS = {
    "label",
    "time",
    "time_index",
    "frame",
    "area",
    "area_px",
    "volume_voxels",
    "sample_id",
    "file_id",
}

_PREFERRED_VALUE_TOKENS = (
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "response_mean",
    "peak_amplitude",
    "auc",
    "duration_above_threshold",
    "area_um2",
    "volume_um3",
)


def default_statistics_value_columns(df: pd.DataFrame, *, limit: int = 6) -> list[str]:
    """Pick measurement-like numeric columns for automatic summaries.

    The selector is intentionally conservative: labels, time indices, raw pixel
    area, centroids, and bookkeeping columns are skipped. Preferred microscopy
    measurement columns are selected first, then any remaining non-bookkeeping
    numeric columns are used as a fallback so reports do not silently omit a
    custom measurement column.
    """
    numeric = list(df.select_dtypes(include="number").columns)

    def usable(col: str) -> bool:
        if col in _EXCLUDED_VALUE_COLUMNS:
            return False
        if col.startswith("centroid") or col.startswith("bbox"):
            return False
        if col.endswith("_baseline"):
            return False
        return True

    out: list[str] = []
    for token in _PREFERRED_VALUE_TOKENS:
        for col in numeric:
            if col not in out and usable(col) and token in col:
                out.append(col)
                if len(out) >= limit:
                    return out
    for col in numeric:
        if col not in out and usable(col):
            out.append(col)
            if len(out) >= limit:
                break
    return out


def _trace_group_columns(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    group_cols: list[str] | None = None,
) -> list[str]:
    if group_cols is not None:
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            raise ValueError(f"group_cols missing from table: {missing}")
        return list(group_cols)
    cols = [
        col
        for col in (
            "sample_name",
            "sample_id",
            "group",
            "file_id",
            "source_layer",
            "image_layer",
            label_col,
        )
        if col in df.columns
    ]
    if label_col not in cols and label_col in df.columns:
        cols.append(label_col)
    if not cols:
        raise ValueError("no trace grouping columns found; pass group_cols explicitly")
    return cols


def _descriptive_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "p5": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "cv": np.nan,
            "outlier_iqr_count": 0,
        }
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(arr))
    q1, q3 = np.percentile(arr, (25, 75))
    iqr = float(q3 - q1)
    if iqr > 0:
        low = float(q1 - 1.5 * iqr)
        high = float(q3 + 1.5 * iqr)
        outliers = int(np.count_nonzero((arr < low) | (arr > high)))
    else:
        outliers = 0
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "sem": float(std / np.sqrt(n)) if n > 1 else 0.0,
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "q1": float(q1),
        "q3": float(q3),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "iqr": iqr,
        "cv": float(std / mean) if mean != 0 and np.isfinite(mean) else np.nan,
        "outlier_iqr_count": outliers,
    }


def _describe_by(df: pd.DataFrame, value_col: str, by: list[str]) -> pd.DataFrame:
    if not by:
        row = _descriptive_stats(df[value_col])
        return pd.DataFrame([{**row}])
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(by, dropna=False, sort=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: key_tuple[i] for i, col in enumerate(by)}
        row.update(_descriptive_stats(group[value_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_weight_col(
    df: pd.DataFrame, weight_col: str | None, value_col: str
) -> str | None:
    """Resolve the sample-aggregation weight column.

    ``"auto"`` → the ``area`` column (object voxel/pixel count from regionprops)
    when it exists and isn't the value being compared, else unweighted. ``None`` /
    ``"none"`` → unweighted. Any other string must name an existing column.
    """
    if weight_col in (None, "none", "None", ""):
        return None
    if weight_col == "auto":
        if "area" in df.columns and value_col != "area":
            return "area"
        return None
    if weight_col not in df.columns:
        raise ValueError(
            f"weight_col {weight_col!r} not found in columns: {list(df.columns)}"
        )
    return weight_col


def _weighted_mean(values: Any, weights: Any) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _sample_level_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    sample_col: str,
    group_col: str,
    weight_col: str | None = None,
) -> pd.DataFrame | None:
    if sample_col not in df.columns:
        return None
    by = [sample_col]
    if group_col in df.columns:
        by.append(group_col)
    grouped = df.groupby(by, dropna=False, sort=False)
    summary = (
        grouped[value_col]
        .agg(n_objects="count", mean="mean", median="median", std="std", sem="sem")
        .reset_index()
    )
    if weight_col is not None:
        weighted = (
            grouped.apply(
                lambda g: _weighted_mean(g[value_col], g[weight_col]),
                include_groups=False,
            )
            .rename("weighted")
            .reset_index()
        )
        summary = summary.merge(weighted, on=by, how="left")
    return summary


def _analysis_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    sample_col: str,
    level: SummaryLevel,
    sample_agg: SampleAgg,
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    warnings: list[str] = []
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in columns: {list(df.columns)}")
    if level in {"auto", "sample"}:
        sample_df = _sample_level_frame(
            df,
            value_col,
            sample_col=sample_col,
            group_col=group_col,
            weight_col=weight_col,
        )
        if sample_df is not None:
            if weight_col is not None and "weighted" in sample_df.columns:
                out = sample_df.rename(columns={"weighted": "analysis_value"}).copy()
                warnings.append(
                    f"per-sample values are area-weighted means (weighted by {weight_col!r}: "
                    "total signal / total area), so small objects contribute in proportion to "
                    "their size; pass weight_col=None for an unweighted per-object mean."
                )
            else:
                out = sample_df.rename(columns={sample_agg: "analysis_value"}).copy()
            return out, "analysis_value", "sample", warnings
        if level == "sample":
            raise ValueError(
                f"sample-level analysis requested, but sample_col {sample_col!r} is absent"
            )
        warnings.append(
            "sample_col was not present; object-level values were used. "
            "For biological group inference, sample-level summaries are preferred."
        )
    out = df.copy()
    out["analysis_value"] = pd.to_numeric(out[value_col], errors="coerce")
    return out, "analysis_value", "object", warnings



def _ordered_group_values(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    reference_group: str | None,
) -> list[tuple[Any, np.ndarray]]:
    rows: list[tuple[Any, np.ndarray]] = []
    for group, part in df.groupby(group_col, dropna=False, sort=False):
        values = pd.to_numeric(part[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            rows.append((group, values))
    rows.sort(key=lambda item: str(item[0]))
    if reference_group is not None:
        ref = str(reference_group)
        rows.sort(key=lambda item: 0 if str(item[0]) == ref else 1)
    return rows


def _bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = float(np.mean(bb) - np.mean(aa))
    lo, hi = np.percentile(diffs, (2.5, 97.5))
    return float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    denom = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((np.mean(b) - np.mean(a)) / denom)


def _hedges_g(d: float, n_total: int) -> float:
    if not np.isfinite(d) or n_total <= 3:
        return np.nan
    return float(d * (1.0 - 3.0 / (4.0 * n_total - 9.0)))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = 0
    less = 0
    for av in a:
        greater += int(np.count_nonzero(b > av))
        less += int(np.count_nonzero(b < av))
    return float((greater - less) / (len(a) * len(b)))


@tool(
    description="Create publication-oriented descriptive statistics for a numeric table "
    "column. Reports mean, median, SD, SEM, IQR, percentiles, CV, and IQR outlier "
    "counts at object level and, when sample_name is present, sample level.",
    phase="7",
    worker=True,
)
def describe_table(
    table_name: str,
    value_col: str,
    group_col: str | None = "group",
    sample_col: str = "sample_name",
    save_csv: bool = True,
) -> dict[str, Any]:
    df, dropped = finite_numeric_frame(get_table(table_name), value_col)
    if df.empty:
        raise ValueError(f"table {table_name!r} has no finite values in {value_col!r}")

    object_by = [group_col] if group_col and group_col in df.columns else []
    object_desc = _describe_by(df, value_col, object_by)
    object_table = put_table(
        f"stats_object__{slugify_result_name(table_name)}__{slugify_result_name(value_col)}",
        object_desc,
        spec={
            "tool": "describe_table",
            "source_table": table_name,
            "value_col": value_col,
            "level": "object",
        },
    )
    object_rows = object_desc.to_dict(orient="records")
    for row in object_rows:
        row.update({
            "value_col": value_col,
            "level": "object",
            "sample_aggregation": "",
        })
    if save_csv:
        register_stats_rows(kind="describe", table=table_name, rows=object_rows)
    object_csv = None  # legacy field; long-format file lives in bundle/stats.

    sample_table: str | None = None
    sample_csv: str | None = None
    sample_rows = 0
    sample_df = None
    if group_col and group_col in df.columns:
        sample_df = _sample_level_frame(
            df,
            value_col,
            sample_col=sample_col,
            group_col=group_col,
        )
    elif sample_col in df.columns:
        sample_df = _sample_level_frame(
            df,
            value_col,
            sample_col=sample_col,
            group_col=str(group_col or "group"),
        )
    if sample_df is not None:
        by = [group_col] if group_col and group_col in sample_df.columns else []
        sample_desc = _describe_by(sample_df.rename(columns={"mean": value_col}), value_col, by)
        sample_table = put_table(
            f"stats_sample__{slugify_result_name(table_name)}__{slugify_result_name(value_col)}",
            sample_desc,
            spec={
                "tool": "describe_table",
                "source_table": table_name,
                "value_col": value_col,
                "level": "sample",
                "sample_aggregation": "mean",
            },
        )
        sample_rows = int(len(sample_df))
        sample_csv = None
        sample_rows_for_csv = sample_desc.to_dict(orient="records")
        for row in sample_rows_for_csv:
            row.update({
                "value_col": value_col,
                "level": "sample",
                "sample_aggregation": "mean",
            })
        if save_csv:
            register_stats_rows(kind="describe", table=table_name, rows=sample_rows_for_csv)

    return {
        "source_table": table_name,
        "value_col": value_col,
        "n_object_rows": int(len(df)),
        "n_sample_rows": sample_rows,
        "dropped_nonfinite": dropped,
        "object_stats_table": object_table,
        "sample_stats_table": sample_table,
        "object_stats_csv": object_csv,
        "sample_stats_csv": sample_csv,
    }


def _signed_rank_biserial(d: np.ndarray) -> float:
    """Rank-biserial correlation for a paired sample, from the signed ranks of the nonzero
    differences: ``r = (R+ - R-) / (R+ + R-)``. Ties get average ranks; zeros are dropped
    (Wilcoxon 'wilcox' zero method). Returns 0.0 when all differences are zero. This is
    unambiguous in sign (positive when ``d = b - a`` skews positive), unlike deriving it from
    SciPy's smaller-rank-sum W."""
    from scipy.stats import rankdata

    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    r_plus = float(ranks[nz > 0].sum())
    r_minus = float(ranks[nz < 0].sum())
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def _bootstrap_mean_paired(d: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference — resample ``d`` itself so
    pairing is preserved (never resample the two arms independently)."""
    d = np.asarray(d, dtype=float)
    if n_bootstrap <= 0 or d.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        means[i] = float(np.mean(rng.choice(d, size=d.size, replace=True)))
    lo, hi = np.percentile(means, (2.5, 97.5))
    return float(lo), float(hi)


def _paired_compare(
    analysis: pd.DataFrame,
    group_col: str,
    analysis_col: str,
    sample_col: str,
    *,
    test: str,
    reference_group: str | None,
    value_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Two-group paired comparison, pairing the groups within each sample.

    Consumes the already sample-aggregated ``analysis`` frame (one row per sample x group),
    pivots to aligned per-sample pairs, drops incomplete/non-finite pairs, and runs a
    signed-rank (``wilcoxon``) or paired-t test with ``d = b - a`` (``group_a`` = the
    reference/baseline). Errors loudly rather than silently averaging duplicates.
    """
    from scipy import stats as scipy_stats

    warnings: list[str] = []
    if sample_col not in analysis.columns:
        raise ValueError(
            f"paired tests need sample-level data with a {sample_col!r} column; use a table "
            "with sample_name (object-level rows cannot be paired)"
        )
    if analysis.duplicated([sample_col, group_col]).any():
        raise ValueError(
            f"paired input has multiple rows per ({sample_col}, {group_col}); aggregate to one "
            "value per sample and group first"
        )
    pivot = analysis.pivot(index=sample_col, columns=group_col, values=analysis_col)
    groups = [str(g) for g in pivot.columns]
    if len(groups) != 2:
        raise ValueError(
            f"paired tests require exactly two groups in {group_col!r}; got {len(groups)}"
        )
    ordered = sorted(groups)
    if reference_group is not None:
        ref = str(reference_group)
        if ref not in groups:
            raise ValueError(f"reference_group {reference_group!r} not among groups {groups}")
        ordered = [ref] + [g for g in ordered if g != ref]
    group_a, group_b = ordered[0], ordered[1]

    pair = pivot[[group_a, group_b]]
    n_total = int(len(pair))
    a = pair[group_a].to_numpy(dtype=float)
    b = pair[group_b].to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n_pairs = int(a.size)
    n_dropped = n_total - n_pairs
    if n_dropped > 0:
        warnings.append(
            f"dropped {n_dropped} sample(s) with an incomplete or non-finite inside/outside pair"
        )
    if n_pairs < 2:
        raise ValueError(f"need at least two complete pairs; got {n_pairs}")
    d = b - a

    if test == "wilcoxon":
        if np.all(d == 0):
            stat, pvalue = 0.0, 1.0
            warnings.append("all paired differences are zero; p-value set to 1.0")
        else:
            stat, pvalue = scipy_stats.wilcoxon(b, a)  # (b, a) so any stat sign tracks b - a
        effect, effect_name = _signed_rank_biserial(d), "rank_biserial"
        test_name = "wilcoxon_signed_rank"
    elif test == "paired_t":
        stat, pvalue = scipy_stats.ttest_rel(b, a)
        sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
        if sd == 0.0:
            effect = 0.0 if float(np.mean(d)) == 0.0 else np.nan
            if not np.isfinite(effect):
                warnings.append("paired differences have zero variance; cohens_dz is undefined")
        else:
            effect = float(np.mean(d) / sd)
        effect_name, test_name = "cohens_dz", "paired_t"
    else:
        raise ValueError(f"unknown paired test {test!r}")

    ci_low, ci_high = _bootstrap_mean_paired(d, n_bootstrap=n_bootstrap, seed=seed)
    row = {
        "test": test_name,
        "requested_test": test,
        "data_level": "sample",
        "value_col": value_col,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_pairs": n_pairs,
        "n_dropped_incomplete": n_dropped,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mean_difference_b_minus_a": float(np.mean(d)),
        "median_difference_b_minus_a": float(np.median(d)),
        "mean_difference_ci95_low": ci_low,
        "mean_difference_ci95_high": ci_high,
        "statistic": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
        effect_name: effect,
    }
    return [row], {group_a: n_pairs, group_b: n_pairs}, warnings


def _finalize_compare(
    rows: list[dict[str, Any]],
    *,
    table_name: str,
    value_col: str,
    group_col: str,
    data_level: str,
    sample_agg: str,
    dropped: int,
    warnings: list[str],
    group_counts: dict[str, int],
    save_csv: bool,
    weighted_by: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    posthoc_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shared tail for compare_groups: persist the result rows and build the return dict."""
    result_df = pd.DataFrame(rows)
    if warnings:
        result_df["warnings"] = "; ".join(warnings)
    if weighted_by is not None:
        result_df["weighted_by"] = weighted_by
    result_table = put_table(
        f"stats_compare__{slugify_result_name(table_name)}__{slugify_result_name(value_col)}",
        result_df,
        spec={
            "tool": "compare_groups",
            "source_table": table_name,
            "value_col": value_col,
            "group_col": group_col,
            "data_level": data_level,
            "sample_agg": sample_agg,
            "weighted_by": weighted_by,
            "dropped_nonfinite": dropped,
        },
    )
    out_rows = result_df.to_dict(orient="records")
    for row in out_rows:
        row["value_col"] = value_col
    if save_csv:
        register_stats_rows(kind="compare", table=table_name, rows=out_rows)

    posthoc_table: str | None = None
    if posthoc_rows:
        posthoc_table = put_table(
            f"stats_posthoc__{slugify_result_name(table_name)}__{slugify_result_name(value_col)}",
            pd.DataFrame(posthoc_rows),
            spec={
                "tool": "compare_groups",
                "kind": "posthoc",
                "source_table": table_name,
                "value_col": value_col,
                "group_col": group_col,
            },
        )
    return {
        "source_table": table_name,
        "value_col": value_col,
        "group_col": group_col,
        "data_level": data_level,
        "sample_agg": sample_agg,
        "weighted_by": weighted_by,
        "test": str(result_df.loc[0, "test"]),
        "p_value": float(result_df.loc[0, "p_value"]),
        "result_table": result_table,
        "csv_path": None,
        "group_counts": group_counts,
        "dropped_nonfinite": dropped,
        "test_selection": diagnostics,
        "posthoc": posthoc_rows,
        "posthoc_table": posthoc_table,
        "warnings": warnings,
    }


def _shapiro_p(values: np.ndarray) -> float | None:
    """Shapiro-Wilk normality p-value, or None when it can't be assessed
    (fewer than 3 finite points, or a constant group)."""
    from scipy import stats as scipy_stats

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(np.ptp(x)) == 0.0:
        return None
    try:
        return float(scipy_stats.shapiro(x).pvalue)
    except Exception:
        return None


def _auto_select_test(
    grouped: list[tuple[Any, np.ndarray]], *, alpha: float = 0.05
) -> tuple[str, dict[str, Any], list[str]]:
    """Pick an independent omnibus test, assumption- and n-aware.

    Consistent philosophy across group counts: parametric (Welch for two groups,
    one-way ANOVA for 3+) unless there is positive evidence against normality
    (a group with n>=3 whose Shapiro-Wilk p < alpha), in which case the matching
    non-parametric test (Mann-Whitney / Kruskal-Wallis) is used. Returns the test
    name, a diagnostics dict, and human-readable notes.
    """
    n_groups = len(grouped)
    sizes = {str(name): int(len(v)) for name, v in grouped}
    shapiro = {str(name): _shapiro_p(v) for name, v in grouped}
    n_min = min((len(v) for _, v in grouped), default=0)
    rejected = [name for name, p in shapiro.items() if p is not None and p < alpha]
    non_normal = len(rejected) > 0

    if n_groups == 2:
        test = "mannwhitney" if non_normal else "welch"
    else:
        test = "kruskal" if non_normal else "anova"

    notes: list[str] = []
    if non_normal:
        notes.append(
            f"Shapiro-Wilk rejects normality (p<{alpha}) in group(s) {rejected}; auto-selected the "
            f"non-parametric {test}. Pass test= to override."
        )
    if n_min < 3:
        notes.append(
            f"smallest group has n={n_min} (<3): normality can't be assessed and power is very low — "
            "interpret the p-value with strong caution (report descriptives / individual points)."
        )
    diagnostics = {
        "auto_selected_test": test,
        "group_sizes": sizes,
        "normality_shapiro_p": {
            k: (round(v, 4) if v is not None else None) for k, v in shapiro.items()
        },
        "n_min": int(n_min),
    }
    return test, diagnostics, notes


def _within_subject_warning(
    analysis: pd.DataFrame, group_col: str, sample_col: str, data_level: str
) -> str | None:
    """Warn when the same sample appears in 2+ groups (a paired / within-subject
    design), which an independent test handles incorrectly."""
    if data_level != "sample" or sample_col not in analysis.columns:
        return None
    spanning = analysis.groupby(sample_col)[group_col].nunique(dropna=False)
    n_spanning = int((spanning > 1).sum())
    if n_spanning < 2:
        return None
    return (
        f"{n_spanning} samples appear in more than one {group_col!r} group — this looks like a "
        "paired / within-subject design (e.g. inside-vs-outside or before/after on the same "
        "specimen). An independent test ignores the pairing; prefer test='wilcoxon' (or "
        "'paired_t') unless the groups really are independent."
    )


def _adjust_pvalues(pvals: list[float], method: str) -> list[float]:
    """Holm-Bonferroni (step-down FWER) or Benjamini-Hochberg (step-up FDR) adjustment."""
    m = len(pvals)
    if m == 0 or method == "none":
        return [min(max(p, 0.0), 1.0) for p in pvals]
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals[idx])
            adj[idx] = min(running, 1.0)
    else:  # fdr_bh
        running = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            running = min(running, pvals[idx] * m / (rank + 1))
            adj[idx] = min(running, 1.0)
    return adj


def _games_howell(grouped: list[tuple[Any, np.ndarray]]) -> list[dict[str, Any]]:
    """Games-Howell post-hoc (parametric, unequal variance). The studentized-range
    p-value already controls the family-wise error rate, so p_adjusted == p_value."""
    from scipy.stats import studentized_range

    k = len(grouped)
    stats_ = [(name, float(np.mean(v)), float(np.var(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) for name, v in grouped]
    rows: list[dict[str, Any]] = []
    for i in range(k):
        for j in range(i + 1, k):
            na, ma, va, ni = stats_[i]
            nb, mb, vb, nj = stats_[j]
            se = float(np.sqrt(va / ni + vb / nj)) if ni and nj else 0.0
            if se == 0.0:
                t, df, p = 0.0, float("nan"), (1.0 if ma == mb else 0.0)
            else:
                t = (ma - mb) / se
                df = (va / ni + vb / nj) ** 2 / (
                    (va / ni) ** 2 / (ni - 1) + (vb / nj) ** 2 / (nj - 1)
                )
                p = float(min(studentized_range.sf(abs(t) * np.sqrt(2), k, df), 1.0))
            rows.append({
                "group_a": na, "group_b": nb, "mean_diff_a_minus_b": ma - mb,
                "statistic": t, "df": df, "p_value": p, "p_adjusted": p,
                "method": "games_howell",
            })
    return rows


def _dunn(grouped: list[tuple[Any, np.ndarray]], correction: str) -> list[dict[str, Any]]:
    """Dunn's post-hoc (non-parametric), using pooled tie-corrected rank variance,
    with Holm/BH adjustment across the pairwise family."""
    from scipy.stats import norm, rankdata

    concat = np.concatenate([v for _, v in grouped])
    n_total = concat.size
    ranks = rankdata(concat)
    _, counts = np.unique(concat, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts)) / (12.0 * (n_total - 1)) if n_total > 1 else 0.0
    var_factor = n_total * (n_total + 1) / 12.0 - tie_term

    mean_ranks: list[float] = []
    sizes: list[int] = []
    pos = 0
    for _name, v in grouped:
        r = ranks[pos: pos + len(v)]
        pos += len(v)
        mean_ranks.append(float(r.mean()))
        sizes.append(len(v))

    k = len(grouped)
    raw: list[float] = []
    pairs: list[tuple[int, int, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            se = float(np.sqrt(var_factor * (1.0 / sizes[i] + 1.0 / sizes[j])))
            z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
            raw.append(float(min(2.0 * norm.sf(abs(z)), 1.0)))
            pairs.append((i, j, z))
    adjusted = _adjust_pvalues(raw, correction)
    return [
        {
            "group_a": grouped[i][0], "group_b": grouped[j][0], "statistic": z,
            "p_value": p, "p_adjusted": pa, "method": f"dunn+{correction}",
        }
        for (i, j, z), p, pa in zip(pairs, raw, adjusted)
    ]


@tool(
    description="Compare groups for a numeric measurement with conservative defaults. "
    "Uses sample-level means when sample_name is present; otherwise object-level "
    "values are used with a warning. test='auto' (default) chooses parametric "
    "(Welch t-test / one-way ANOVA) vs non-parametric (Mann-Whitney U / Kruskal-Wallis) "
    "from a Shapiro-Wilk normality check — consistently across 2 and 3+ groups — warns "
    "when a group has n<3, and flags a likely paired design (the same sample appearing in "
    "two groups). Also supports paired Wilcoxon signed-rank / paired-t "
    "(test='wilcoxon' / 'paired_t') for within-sample designs such as "
    "inside-vs-outside a domain measured on the same specimen. When aggregating "
    "objects to a per-sample value, weight_col='auto' (default) weights by the `area` "
    "column when present — i.e. total signal / total area, so small debris objects "
    "count in proportion to their size rather than one-object-one-vote; pass "
    "weight_col=None for a plain per-object mean (e.g. per-cell designs). For 3+ groups "
    "it also runs a corrected post-hoc pairwise test (Games-Howell for ANOVA; Dunn's "
    "with Holm correction for Kruskal), returned under `posthoc` — so which pairs differ "
    "is answered without uncorrected multiple comparisons.",
    phase="7",
    worker=True,
)
def compare_groups(
    table_name: str,
    value_col: str,
    group_col: str = "group",
    sample_col: str = "sample_name",
    level: Literal["auto", "sample", "object"] = "auto",
    sample_agg: Literal["mean", "median"] = "mean",
    weight_col: str | None = "auto",
    test: Literal[
        "auto", "ttest", "welch", "mannwhitney", "anova", "kruskal", "wilcoxon", "paired_t"
    ] = "auto",
    reference_group: str | None = None,
    posthoc: bool = True,
    posthoc_correction: Literal["holm", "fdr_bh", "none"] = "holm",
    n_bootstrap: int = 5000,
    seed: int = 12345,
    save_csv: bool = True,
) -> dict[str, Any]:
    from scipy import stats as scipy_stats

    df, dropped = finite_numeric_frame(get_table(table_name), value_col)
    weight = resolve_weight_col(df, weight_col, value_col)
    analysis, analysis_col, data_level, warnings = _analysis_frame(
        df,
        value_col,
        group_col=group_col,
        sample_col=sample_col,
        level=level,
        sample_agg=sample_agg,
        weight_col=weight,
    )
    # weighting only applies to sample-level aggregation
    weighted_by = weight if data_level == "sample" else None
    if test in ("wilcoxon", "paired_t"):
        rows, group_counts, paired_warnings = _paired_compare(
            analysis,
            group_col,
            analysis_col,
            sample_col,
            test=test,
            reference_group=reference_group,
            value_col=value_col,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return _finalize_compare(
            rows,
            table_name=table_name,
            value_col=value_col,
            group_col=group_col,
            data_level=data_level,
            sample_agg=sample_agg,
            dropped=dropped,
            warnings=warnings + paired_warnings,
            group_counts=group_counts,
            save_csv=save_csv,
            weighted_by=weighted_by,
        )

    if (
        data_level == "object"
        and sample_col in df.columns
        and df.groupby([sample_col, group_col]).size().gt(1).any()
    ):
        warnings.append(
            "object-level rows are nested within samples (multiple objects per sample and "
            "group); this independent test risks pseudoreplication — prefer sample-level "
            "aggregation and a paired test for cross-specimen inference"
        )

    grouped = _ordered_group_values(analysis, group_col, analysis_col, reference_group)
    if len(grouped) < 2:
        raise ValueError(f"need at least two groups in {group_col!r}; got {len(grouped)}")

    n_groups = len(grouped)
    requested_test = test
    diagnostics: dict[str, Any] | None = None
    if test == "auto":
        test, diagnostics, auto_notes = _auto_select_test(grouped)
        warnings.extend(auto_notes)
    paired_warning = _within_subject_warning(analysis, group_col, sample_col, data_level)
    if paired_warning:
        warnings.append(paired_warning)

    rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] | None = None
    group_counts = {str(name): int(len(values)) for name, values in grouped}
    object_counts = (
        df.groupby(group_col, dropna=False)[value_col].size().to_dict()
        if group_col in df.columns
        else {}
    )

    if n_groups == 2:
        (name_a, a), (name_b, b) = grouped
        if len(a) < 2 or len(b) < 2:
            warnings.append("one or more groups have fewer than two analysis units")
        zero_variance = (
            (len(a) > 1 and float(np.var(a, ddof=1)) == 0.0)
            or (len(b) > 1 and float(np.var(b, ddof=1)) == 0.0)
        )
        if zero_variance:
            warnings.append(
                "one or more groups have zero within-group variance; parametric "
                "p-values should be interpreted cautiously"
            )
        if test in {"ttest", "welch"}:
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="Precision loss occurred",
                    category=RuntimeWarning,
                )
                stat, pvalue = scipy_stats.ttest_ind(
                    a,
                    b,
                    equal_var=False,
                    nan_policy="omit",
                )
            test_name = "welch_ttest"
        elif test == "mannwhitney":
            stat, pvalue = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "mann_whitney_u"
        else:
            raise ValueError("two-group tests must be auto, ttest/welch, or mannwhitney")
        if not np.isfinite(pvalue):
            if len(a) and len(b) and np.all(a == a[0]) and np.all(b == b[0]) and np.isclose(a[0], b[0]):
                stat = 0.0
                pvalue = 1.0
                warnings.append(
                    "both groups were identical constants; p-value was set to 1.0"
                )
        mean_diff = float(np.mean(b) - np.mean(a))
        median_diff = float(np.median(b) - np.median(a))
        ci_low, ci_high = _bootstrap_mean_difference(
            a,
            b,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        d = _cohens_d(a, b)
        row = {
            "test": test_name,
            "requested_test": requested_test,
            "data_level": data_level,
            "value_col": value_col,
            "group_col": group_col,
            "group_a": name_a,
            "group_b": name_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "object_n_a": int(object_counts.get(name_a, object_counts.get(str(name_a), len(a)))),
            "object_n_b": int(object_counts.get(name_b, object_counts.get(str(name_b), len(b)))),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "mean_difference_b_minus_a": mean_diff,
            "median_difference_b_minus_a": median_diff,
            "mean_difference_ci95_low": ci_low,
            "mean_difference_ci95_high": ci_high,
            "statistic": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
            "cohens_d": d,
            "hedges_g": _hedges_g(d, len(a) + len(b)),
            "cliffs_delta": _cliffs_delta(a, b),
        }
        rows.append(row)
    else:
        values = [vals for _name, vals in grouped]
        if test == "anova":
            stat, pvalue = scipy_stats.f_oneway(*values)
            test_name = "one_way_anova"
            all_values = np.concatenate(values)
            grand_mean = float(np.mean(all_values))
            ss_between = sum(len(v) * float((np.mean(v) - grand_mean) ** 2) for v in values)
            ss_total = float(np.sum((all_values - grand_mean) ** 2))
            effect = float(ss_between / ss_total) if ss_total > 0 else np.nan
            effect_name = "eta_squared"
        elif test in {"auto", "kruskal"}:
            stat, pvalue = scipy_stats.kruskal(*values)
            test_name = "kruskal_wallis"
            total_n = sum(len(v) for v in values)
            effect = float((float(stat) - n_groups + 1) / (total_n - n_groups)) if total_n > n_groups else np.nan
            effect_name = "epsilon_squared"
        else:
            raise ValueError("multi-group tests must be auto, anova, or kruskal")
        row = {
            "test": test_name,
            "requested_test": requested_test,
            "data_level": data_level,
            "value_col": value_col,
            "group_col": group_col,
            "n_groups": n_groups,
            "groups": ";".join(str(name) for name, _values in grouped),
            "analysis_n_total": int(sum(len(v) for v in values)),
            "object_n_total": int(len(df)),
            "statistic": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
            effect_name: effect,
        }
        rows.append(row)

        if posthoc and n_groups >= 3:
            try:
                posthoc_rows = (
                    _games_howell(grouped)
                    if test_name == "one_way_anova"
                    else _dunn(grouped, posthoc_correction)
                )
            except Exception as exc:  # noqa: BLE001 - post-hoc is advisory, never fatal
                warnings.append(
                    f"post-hoc pairwise test skipped: {type(exc).__name__}: {exc}"
                )

    return _finalize_compare(
        rows,
        table_name=table_name,
        value_col=value_col,
        group_col=group_col,
        data_level=data_level,
        sample_agg=sample_agg,
        dropped=dropped,
        warnings=warnings,
        group_counts=group_counts,
        save_csv=save_csv,
        weighted_by=weighted_by,
        diagnostics=diagnostics,
        posthoc_rows=posthoc_rows,
    )


def _existing_auto_statistics_keys() -> set[tuple[str, str, str]]:

    keys: set[tuple[str, str, str]] = set()
    covered_inputs: dict[str, tuple[list[str], str]] = {}
    for name in list_tables():
        try:
            spec = dict(get_table_entry(name).spec or {})
        except KeyError:
            continue
        if spec.get("tool") == "batch_auto_statistics_input":
            sources = [str(s) for s in (spec.get("source_tables") or [])]
            value_col = spec.get("value_col")
            if value_col:
                covered_inputs[name] = (sources, str(value_col))
        source = spec.get("source_table")
        value_col = spec.get("value_col")
        tool_name = spec.get("tool")
        if tool_name == "describe_table" and source and value_col:
            keys.add((str(source), str(value_col), "describe"))
        elif tool_name == "compare_groups" and source and value_col:
            keys.add((str(source), str(value_col), "compare"))

    # Batch stats operate on temporary stats_input tables. Treat those as
    # covering their primary measurement tables to avoid duplicate report stats.
    for stats_input_name, (sources, value_col) in covered_inputs.items():
        for source in sources:
            for kind in ("describe", "compare"):
                if (stats_input_name, value_col, kind) in keys:
                    keys.add((source, value_col, kind))
    return keys


def _is_report_measurement_table(name: str, spec: dict[str, Any]) -> bool:
    tool_name = str(spec.get("tool") or "")
    op_name = str(spec.get("op") or "")
    if name.startswith(("stats_", "summary_", "plotdata_")):
        return False
    if tool_name in _NON_MEASUREMENT_TABLE_TOOLS:
        return False
    if op_name in _NON_MEASUREMENT_TABLE_OPS:
        return False
    return True


def _table_has_comparable_groups(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
) -> bool:
    return group_col in df.columns and df[group_col].nunique(dropna=True) >= 2


def _ensure_statistics_for_column(
    table_name: str,
    value_col: str,
    valid: pd.DataFrame,
    existing: set[tuple[str, str, str]],
    *,
    save_csv: bool,
    require_group_for_comparison: bool,
) -> dict[str, Any] | None:
    desc: dict[str, Any] = {}
    created_any = False
    if (table_name, value_col, "describe") not in existing:
        desc = describe_table(table_name, value_col, save_csv=save_csv)
        existing.add((table_name, value_col, "describe"))
        created_any = True

    compare: dict[str, Any] | None = None
    compare_error: str | None = None
    if (
        _table_has_comparable_groups(valid)
        and (table_name, value_col, "compare") not in existing
    ):
        try:
            compare = compare_groups(table_name, value_col, save_csv=save_csv)
            existing.add((table_name, value_col, "compare"))
            created_any = True
        except Exception as exc:  # noqa: BLE001
            compare_error = f"{type(exc).__name__}: {exc}"
            created_any = True
    elif (
        require_group_for_comparison
        and (table_name, value_col, "compare") not in existing
    ):
        compare_error = "group comparison skipped: fewer than two groups"
        created_any = True

    if not created_any:
        return None
    return {
        "source_table": table_name,
        "value_col": value_col,
        "object_stats_table": desc.get("object_stats_table"),
        "sample_stats_table": desc.get("sample_stats_table"),
        "comparison_table": (
            compare.get("result_table")
            if isinstance(compare, dict)
            else None
        ),
        "comparison_p_value": (
            compare.get("p_value")
            if isinstance(compare, dict)
            else None
        ),
        "comparison_error": compare_error,
    }


def ensure_default_statistics(
    *,
    save_csv: bool = True,
    max_value_columns: int = 6,
    require_group_for_comparison: bool = True,
) -> list[dict[str, Any]]:
    """Create missing report-ready summary and comparison tables.

    Reports are often generated after measurement but before the user explicitly
    calls `describe_table` or `compare_groups`. This helper scans current
    measurement-like tables, writes descriptive summaries for selected numeric
    columns, and runs group comparisons when a table has at least two groups.
    Existing stats tables are respected, so repeated report generation does not
    keep duplicating outputs.
    """

    existing = _existing_auto_statistics_keys()
    outputs: list[dict[str, Any]] = []
    table_names = list(list_tables())
    for table_name in table_names:
        try:
            entry = get_table_entry(table_name)
        except KeyError:
            continue
        spec = dict(entry.spec or {})
        if not _is_report_measurement_table(table_name, spec):
            continue
        df = entry.df
        if df is None or df.empty:
            continue
        value_cols = default_statistics_value_columns(
            df,
            limit=int(max_value_columns),
        )
        for value_col in value_cols:
            valid, _dropped = finite_numeric_frame(df, value_col, missing="empty")
            if valid.empty:
                continue
            output = _ensure_statistics_for_column(
                table_name,
                value_col,
                valid,
                existing,
                save_csv=save_csv,
                require_group_for_comparison=require_group_for_comparison,
            )
            if output is not None:
                outputs.append(output)
    return outputs


@tool(
    description="Normalize ROI/cell intensity time courses without overwriting raw "
    "measurements. Supports raw, baseline subtraction, F/F0, DeltaF/F0, z-score, "
    "and min-max visualization normalization. F0 is computed per trace.",
    phase="7",
    worker=True,
)
def normalize_timecourse(
    table_name: str,
    value_col: str = "mean_intensity",
    method: Literal[
        "raw",
        "baseline_subtract",
        "f_over_f0",
        "delta_f_over_f0",
        "f_over_f0_rolling",
        "delta_f_over_f0_rolling",
        "zscore",
        "minmax",
    ] = "delta_f_over_f0",
    baseline: tuple[float, float] | None = None,
    f0_window: int | None = None,
    f0_percentile: float = 10.0,
    group_cols: list[str] | None = None,
    time_col: str | None = None,
    label_col: str = "label",
    output_col: str | None = None,
    new_table_name: str | None = None,
) -> dict[str, Any]:
    df, dropped = finite_numeric_frame(get_table(table_name), value_col)
    tcol = infer_time_column(df, time_col)
    trace_cols = _trace_group_columns(df, label_col=label_col, group_cols=group_cols)
    out_col = output_col or {
        "raw": value_col,
        "baseline_subtract": f"{value_col}_baseline_subtracted",
        "f_over_f0": f"{value_col}_f_over_f0",
        "delta_f_over_f0": f"{value_col}_delta_f_over_f0",
        "f_over_f0_rolling": f"{value_col}_f_over_f0_rolling",
        "delta_f_over_f0_rolling": f"{value_col}_delta_f_over_f0_rolling",
        "zscore": f"{value_col}_zscore",
        "minmax": f"{value_col}_minmax",
    }[method]
    out = df.copy()
    out[out_col] = np.nan
    out[f"{out_col}_baseline"] = np.nan
    warnings: list[str] = []

    rolling_methods = {"f_over_f0_rolling", "delta_f_over_f0_rolling"}

    def _rolling_f0(values: np.ndarray, window: int, pct: float) -> np.ndarray:
        n = len(values)
        w = max(3, int(window) | 1)
        half = w // 2
        out_f0 = np.empty(n, dtype=float)
        for i in range(n):
            seg = values[max(0, i - half): min(n, i + half + 1)]
            seg = seg[np.isfinite(seg)]
            out_f0[i] = np.nanpercentile(seg, pct) if seg.size else np.nan
        return out_f0

    if method in rolling_methods:
        baseline_times: set[float] = set()
    elif baseline is None:
        unique_times = np.asarray(sorted(pd.unique(df[tcol])), dtype=float)
        n_base = max(1, int(np.ceil(len(unique_times) * 0.1)))
        baseline_times = set(unique_times[:n_base].tolist())
        warnings.append(
            "baseline was not provided; the earliest 10% of timepoints "
            f"({n_base} point(s)) were used as F0"
        )
    else:
        start, end = float(baseline[0]), float(baseline[1])
        baseline_times = set(
            pd.to_numeric(df.loc[(df[tcol] >= start) & (df[tcol] <= end), tcol], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        if not baseline_times:
            raise ValueError(f"baseline window {baseline!r} contains no rows")

    bad_baseline = 0
    for _key, group in out.groupby(trace_cols, dropna=False, sort=False):
        idx = group.index
        vals = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        if method in rolling_methods:
            window = f0_window or max(3, (int(round(len(vals) * 0.1)) | 1))
            f0_series = _rolling_f0(vals, window, f0_percentile)
            out.loc[idx, f"{out_col}_baseline"] = f0_series
            safe = np.where(f0_series != 0, f0_series, np.nan)
            norm = vals / safe if method == "f_over_f0_rolling" else (vals - safe) / safe
            out.loc[idx, out_col] = norm
            continue
        times = pd.to_numeric(group[tcol], errors="coerce").to_numpy(dtype=float)
        base_mask = np.asarray([t in baseline_times for t in times], dtype=bool)
        base_vals = vals[base_mask & np.isfinite(vals)]
        if base_vals.size == 0:
            bad_baseline += 1
            continue
        f0 = float(np.mean(base_vals))
        out.loc[idx, f"{out_col}_baseline"] = f0
        if method == "raw":
            norm = vals
        elif method == "baseline_subtract":
            norm = vals - f0
        elif method == "f_over_f0":
            norm = vals / f0 if f0 != 0 else np.full_like(vals, np.nan, dtype=float)
        elif method == "delta_f_over_f0":
            norm = (vals - f0) / f0 if f0 != 0 else np.full_like(vals, np.nan, dtype=float)
        elif method == "zscore":
            sd = float(np.std(base_vals, ddof=1)) if base_vals.size > 1 else 0.0
            norm = (vals - f0) / sd if sd > 0 else np.full_like(vals, np.nan, dtype=float)
        else:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            norm = (vals - lo) / (hi - lo) if hi > lo else np.full_like(vals, np.nan, dtype=float)
        out.loc[idx, out_col] = norm
    if bad_baseline:
        warnings.append(f"{bad_baseline} trace(s) had no usable baseline and were set to NaN")

    result_table = put_table(
        new_table_name or f"{table_name}_{method}",
        out,
        spec={
            "tool": "normalize_timecourse",
            "source_table": table_name,
            "value_col": value_col,
            "output_col": out_col,
            "method": method,
            "baseline": baseline,
            "time_col": tcol,
            "group_cols": trace_cols,
        },
    )
    return {
        "source_table": table_name,
        "table_name": result_table,
        "value_col": value_col,
        "output_col": out_col,
        "method": method,
        "time_col": tcol,
        "group_cols": trace_cols,
        "dropped_nonfinite": dropped,
        "warnings": warnings,
    }


@tool(
    description="Extract response features from normalized or raw time-course traces. "
    "Produces one row per ROI/cell trace with baseline mean, response mean, peak, "
    "time to peak, AUC, and optional duration above threshold.",
    phase="7",
    worker=True,
)
def extract_timecourse_features(
    table_name: str,
    value_col: str,
    baseline_window: tuple[float, float] | None = None,
    response_window: tuple[float, float] | None = None,
    threshold: float | None = None,
    group_cols: list[str] | None = None,
    time_col: str | None = None,
    label_col: str = "label",
    new_table_name: str | None = None,
    save_csv: bool = True,
) -> dict[str, Any]:
    df, dropped = finite_numeric_frame(get_table(table_name), value_col)
    tcol = infer_time_column(df, time_col)
    trace_cols = _trace_group_columns(df, label_col=label_col, group_cols=group_cols)
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(trace_cols, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: key_tuple[i] for i, col in enumerate(trace_cols)}
        ordered = group.sort_values(tcol)
        times = pd.to_numeric(ordered[tcol], errors="coerce").to_numpy(dtype=float)
        vals = pd.to_numeric(ordered[value_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(times) & np.isfinite(vals)
        times = times[finite]
        vals = vals[finite]
        if vals.size == 0:
            continue
        base_vals = vals
        if baseline_window is not None:
            b0, b1 = float(baseline_window[0]), float(baseline_window[1])
            base_vals = vals[(times >= b0) & (times <= b1)]
        resp_vals = vals
        resp_times = times
        if response_window is not None:
            r0, r1 = float(response_window[0]), float(response_window[1])
            mask = (times >= r0) & (times <= r1)
            resp_vals = vals[mask]
            resp_times = times[mask]
        if resp_vals.size == 0:
            continue
        peak_idx = int(np.argmax(resp_vals))
        row.update(
            {
                "n_timepoints": int(vals.size),
                "baseline_mean": float(np.mean(base_vals)) if base_vals.size else np.nan,
                "response_mean": float(np.mean(resp_vals)),
                "response_median": float(np.median(resp_vals)),
                "peak_amplitude": float(resp_vals[peak_idx]),
                "time_to_peak": float(resp_times[peak_idx]),
                "auc": float(np.trapezoid(resp_vals, resp_times)) if resp_vals.size > 1 else 0.0,
            }
        )
        if threshold is not None:
            above = resp_vals > float(threshold)
            row["duration_above_threshold"] = float(np.trapezoid(above.astype(float), resp_times)) if above.size > 1 else float(above.any())
            row["n_points_above_threshold"] = int(np.count_nonzero(above))
        rows.append(row)
    features = pd.DataFrame(rows)
    result_table = put_table(
        new_table_name or f"{table_name}_features",
        features,
        spec={
            "tool": "extract_timecourse_features",
            "source_table": table_name,
            "value_col": value_col,
            "baseline_window": baseline_window,
            "response_window": response_window,
            "threshold": threshold,
            "time_col": tcol,
            "group_cols": trace_cols,
        },
    )
    tc_rows = features.to_dict(orient="records")
    for row in tc_rows:
        row["value_col"] = value_col
    if save_csv:
        register_stats_rows(kind="timecourse_features", table=table_name, rows=tc_rows)
    csv_path = None
    return {
        "source_table": table_name,
        "table_name": result_table,
        "csv_path": csv_path,
        "value_col": value_col,
        "time_col": tcol,
        "n_traces": int(len(features)),
        "dropped_nonfinite": dropped,
        "columns": list(features.columns),
    }
