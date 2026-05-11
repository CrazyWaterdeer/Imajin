from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from imajin.agent.state import get_table, put_table
from imajin.paths import normalize_user_path
from imajin.results import record_result, slugify_result_name, unique_result_path
from imajin.tools._dataframes import finite_numeric_frame, infer_time_column
from imajin.tools.registry import tool


_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _figure_path(stem: str, output_path: str | None, fmt: str) -> Path:
    suffix = fmt.lower().lstrip(".")
    if output_path:
        out = normalize_user_path(output_path).resolve()
        if not out.suffix:
            out = out.with_suffix(f".{suffix}")
        return out
    try:
        from imajin.result_bundles import current_bundle

        bundle = current_bundle()
    except Exception:
        bundle = None
    filename = f"{slugify_result_name(stem)}.{suffix}"
    if bundle is not None:
        return Path(bundle) / "figures" / filename
    return unique_result_path("figures", filename)


def _save_figure(fig: Any, out: Path, *, dpi: int, metadata: dict[str, Any]) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", transparent=False)
    record_result("figure", out, metadata)
    return str(out)


def _sample_or_object_values(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    sample_col: str,
    level: Literal["auto", "sample", "object"],
    sample_agg: Literal["mean", "median"],
) -> tuple[pd.DataFrame, str]:
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in columns: {list(df.columns)}")
    if level in {"auto", "sample"} and sample_col in df.columns:
        plot_df = (
            df.groupby([sample_col, group_col], dropna=False, sort=False)[value_col]
            .agg(n_objects="count", mean="mean", median="median")
            .reset_index()
        )
        plot_df["plot_value"] = plot_df[sample_agg]
        return plot_df, "sample"
    if level == "sample":
        raise ValueError(f"sample-level plot requested, but {sample_col!r} is absent")
    plot_df = df.copy()
    plot_df["plot_value"] = plot_df[value_col]
    return plot_df, "object"


def _style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def _format_p_value(p_value: float | None) -> str | None:
    if p_value is None:
        return None
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(p):
        return None
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def _p_stars(p_value: float | None) -> str | None:
    if p_value is None:
        return None
    p = float(p_value)
    if not np.isfinite(p):
        return None
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _annotate_p_value(
    ax: Any,
    *,
    positions: np.ndarray,
    values: list[np.ndarray],
    p_value: float | None,
    label: str | None,
) -> None:
    text = _p_stars(p_value)
    if text is None:
        text = label
    if not text:
        return
    finite_parts = [v[np.isfinite(v)] for v in values if len(v)]
    if not finite_parts:
        return
    finite_values = np.concatenate(finite_parts)
    if finite_values.size == 0:
        return
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1.0)
    y = y_max + 0.10 * y_range
    h = 0.04 * y_range
    x0 = float(positions[0])
    x1 = float(positions[-1])
    if len(positions) == 2:
        ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="#222222", linewidth=0.8)
        ax.text((x0 + x1) / 2.0, y + h, text, ha="center", va="bottom", fontsize=7)
    else:
        ax.text(
            0.5,
            0.98,
            label or text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
    ax.set_ylim(top=y + 3 * h)


def _distribution_groups(
    plot_df: pd.DataFrame,
    *,
    group_col: str,
) -> tuple[list[Any], list[np.ndarray]]:
    groups = [g for g in pd.unique(plot_df[group_col])]
    values = [
        pd.to_numeric(plot_df.loc[plot_df[group_col] == g, "plot_value"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for g in groups
    ]
    return groups, values


def _distribution_statistics(
    table_name: str,
    value_col: str,
    *,
    group_col: str,
    sample_col: str,
    level: Literal["auto", "sample", "object"],
    sample_agg: Literal["mean", "median"],
    stats_test: Literal["auto", "ttest", "welch", "mannwhitney", "anova", "kruskal"],
    show_stats: bool,
    n_groups: int,
) -> tuple[dict[str, Any] | None, float | None, str | None, str | None]:
    if not show_stats or n_groups < 2:
        return None, None, None, None
    try:
        from imajin.tools import stats as _stats

        stats_result = _stats.compare_groups(
            table_name,
            value_col,
            group_col=group_col,
            sample_col=sample_col,
            level=level,
            sample_agg=sample_agg,
            test=stats_test,
            save_csv=True,
        )
        p_value = float(stats_result["p_value"])
        test_name = str(stats_result.get("test") or stats_test)
        p_text = _format_p_value(p_value)
        p_label = f"{test_name}, {p_text}" if p_text else test_name
        return stats_result, p_value, p_label, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"{type(exc).__name__}: {exc}"


@tool(
    description="Export a publication-style group distribution figure from a numeric "
    "measurement table. Defaults to sample-level means when sample_name is present, "
    "with boxplot and jittered points. Saves SVG/PDF/PNG into figures/.",
    phase="7",
    worker=True,
)
def plot_group_distribution(
    table_name: str,
    value_col: str,
    group_col: str = "group",
    sample_col: str = "sample_name",
    level: Literal["auto", "sample", "object"] = "auto",
    sample_agg: Literal["mean", "median"] = "mean",
    output_path: str | None = None,
    format: Literal["svg", "pdf", "png"] = "svg",
    title: str | None = None,
    ylabel: str | None = None,
    width: float = 3.2,
    height: float = 2.6,
    dpi: int = 600,
    show_n: bool = True,
    show_stats: bool = True,
    stats_test: Literal["auto", "ttest", "welch", "mannwhitney", "anova", "kruskal"] = "auto",
    store_plot_data: bool = True,
) -> dict[str, Any]:
    df, _dropped = finite_numeric_frame(get_table(table_name), value_col)
    if df.empty:
        raise ValueError(f"table {table_name!r} has no finite values in {value_col!r}")
    plot_df, data_level = _sample_or_object_values(
        df,
        value_col,
        group_col=group_col,
        sample_col=sample_col,
        level=level,
        sample_agg=sample_agg,
    )
    groups, values = _distribution_groups(plot_df, group_col=group_col)
    stats_result, p_value, p_label, stats_error = _distribution_statistics(
        table_name,
        value_col,
        group_col=group_col,
        sample_col=sample_col,
        level=level,
        sample_agg=sample_agg,
        stats_test=stats_test,
        show_stats=show_stats,
        n_groups=len(groups),
    )

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(float(width), float(height)))
    positions = np.arange(1, len(groups) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 0.9},
        capprops={"color": "#333333", "linewidth": 0.9},
        boxprops={"color": "#333333", "linewidth": 0.9},
    )
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
        patch.set_alpha(0.18)

    rng = np.random.default_rng(12345)
    for i, arr in enumerate(values):
        if arr.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=arr.size)
        ax.scatter(
            np.full(arr.size, positions[i], dtype=float) + jitter,
            arr,
            s=22 if data_level == "sample" else 10,
            alpha=0.9 if data_level == "sample" else 0.45,
            color=_PALETTE[i % len(_PALETTE)],
            edgecolor="#222222" if data_level == "sample" else "none",
            linewidth=0.35,
            zorder=3,
        )
        mean = float(np.mean(arr))
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        ax.errorbar(
            positions[i],
            mean,
            yerr=1.96 * sem,
            fmt="D",
            color="#111111",
            markersize=3.5,
            linewidth=0.9,
            capsize=2,
            zorder=4,
        )

    ax.set_xticks(positions)
    if show_n:
        ax.set_xticklabels(
            [f"{g}\nn={len(v)}" for g, v in zip(groups, values, strict=False)],
            rotation=0,
            ha="center",
        )
    else:
        ax.set_xticklabels([str(g) for g in groups], rotation=25, ha="right")
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    if p_label:
        _annotate_p_value(
            ax,
            positions=positions,
            values=values,
            p_value=p_value,
            label=p_label,
        )
    _style_axes(ax)
    fig.tight_layout()

    out = _figure_path(f"{table_name}__{value_col}__distribution", output_path, format)
    path = _save_figure(
        fig,
        out,
        dpi=dpi,
        metadata={
            "tool": "plot_group_distribution",
            "table_name": table_name,
            "value_col": value_col,
            "data_level": data_level,
            "stats_test": stats_result.get("test") if stats_result else None,
            "p_value": p_value,
            "p_label": p_label,
            "stats_error": stats_error,
        },
    )
    plt.close(fig)

    plot_table = None
    if store_plot_data:
        stored = plot_df.copy()
        stored["data_level"] = data_level
        plot_table = put_table(
            f"plotdata_distribution__{value_col}",
            stored,
            spec={
                "tool": "plot_group_distribution",
                "source_table": table_name,
                "value_col": value_col,
                "group_col": group_col,
                "data_level": data_level,
            },
        )

    return {
        "path": path,
        "format": format,
        "table_name": table_name,
        "value_col": value_col,
        "group_col": group_col,
        "groups": [str(g) for g in groups],
        "data_level": data_level,
        "n_points": int(sum(len(v) for v in values)),
        "plot_data_table": plot_table,
        "stats_test": stats_result.get("test") if stats_result else None,
        "p_value": p_value,
        "p_label": p_label,
        "stats_result_table": stats_result.get("result_table") if stats_result else None,
        "stats_error": stats_error,
    }


@tool(
    description="Export a publication-style time-course figure. With sample_name "
    "present, group means and ribbons are computed across sample-level time traces; "
    "otherwise traces are summarized across ROI/cell labels.",
    phase="7",
    worker=True,
)
def plot_timecourse(
    table_name: str,
    value_col: str = "mean_intensity",
    group_col: str = "group",
    sample_col: str = "sample_name",
    label_col: str = "label",
    time_col: str | None = None,
    output_path: str | None = None,
    format: Literal["svg", "pdf", "png"] = "svg",
    title: str | None = None,
    ylabel: str | None = None,
    width: float = 3.6,
    height: float = 2.6,
    dpi: int = 600,
    interval: Literal["sem", "ci95", "none"] = "sem",
    show_individual: bool = True,
    max_individual_traces: int = 150,
    store_plot_data: bool = True,
) -> dict[str, Any]:
    df, _dropped = finite_numeric_frame(get_table(table_name), value_col)
    if df.empty:
        raise ValueError(f"table {table_name!r} has no finite values in {value_col!r}")
    tcol = infer_time_column(df, time_col)
    if group_col not in df.columns:
        df[group_col] = "all"

    if sample_col in df.columns:
        unit_cols = [group_col, sample_col, tcol]
        unit_level = "sample"
    elif label_col in df.columns:
        unit_cols = [group_col, label_col, tcol]
        unit_level = "roi"
    else:
        unit_cols = [group_col, tcol]
        unit_level = "row"

    unit_df = (
        df.groupby(unit_cols, dropna=False, sort=False)[value_col]
        .mean()
        .reset_index(name="trace_value")
    )
    summary = (
        unit_df.groupby([group_col, tcol], dropna=False, sort=False)["trace_value"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["ci95"] = 1.96 * summary["sem"]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(float(width), float(height)))
    groups = [g for g in pd.unique(summary[group_col])]
    group_ns = {
        group: int(summary.loc[summary[group_col] == group, "n"].max())
        for group in groups
    }
    trace_id_cols = [c for c in unit_cols if c != tcol]
    if show_individual and trace_id_cols:
        traces = list(unit_df.groupby(trace_id_cols, dropna=False, sort=False))
        for i, (_key, trace) in enumerate(traces[: max(0, int(max_individual_traces))]):
            group_value = trace[group_col].iloc[0]
            color = _PALETTE[groups.index(group_value) % len(_PALETTE)] if group_value in groups else "#888888"
            ordered = trace.sort_values(tcol)
            ax.plot(
                ordered[tcol],
                ordered["trace_value"],
                color=color,
                alpha=0.12,
                linewidth=0.55,
                zorder=1,
            )

    for i, group in enumerate(groups):
        part = summary[summary[group_col] == group].sort_values(tcol)
        x = pd.to_numeric(part[tcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(part["mean"], errors="coerce").to_numpy(dtype=float)
        color = _PALETTE[i % len(_PALETTE)]
        label = f"{group} (n={group_ns.get(group, 0)})"
        ax.plot(x, y, color=color, linewidth=1.6, label=label, zorder=3)
        if interval != "none":
            err = pd.to_numeric(part[interval], errors="coerce").fillna(0).to_numpy(dtype=float)
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.18, linewidth=0, zorder=2)

    ax.set_xlabel(tcol)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    if len(groups) > 1:
        ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()

    out = _figure_path(f"{table_name}__{value_col}__timecourse", output_path, format)
    path = _save_figure(
        fig,
        out,
        dpi=dpi,
        metadata={
            "tool": "plot_timecourse",
            "table_name": table_name,
            "value_col": value_col,
            "time_col": tcol,
            "unit_level": unit_level,
        },
    )
    plt.close(fig)

    plot_table = None
    if store_plot_data:
        plot_table = put_table(
            f"plotdata_timecourse__{value_col}",
            summary,
            spec={
                "tool": "plot_timecourse",
                "source_table": table_name,
                "value_col": value_col,
                "time_col": tcol,
                "unit_level": unit_level,
            },
        )

    return {
        "path": path,
        "format": format,
        "table_name": table_name,
        "value_col": value_col,
        "time_col": tcol,
        "group_col": group_col,
        "groups": [str(g) for g in groups],
        "unit_level": unit_level,
        "n_summary_rows": int(len(summary)),
        "plot_data_table": plot_table,
    }


@tool(
    description="Export a publication-style scatter plot for two numeric columns, "
    "useful for channel-intensity relationships or per-cell colocalization tables.",
    phase="7",
    worker=True,
)
def plot_scatter(
    table_name: str,
    x_col: str,
    y_col: str,
    group_col: str | None = "group",
    output_path: str | None = None,
    format: Literal["svg", "pdf", "png"] = "svg",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: float = 3.0,
    height: float = 2.8,
    dpi: int = 600,
    log10: bool = False,
    fit_line: bool = True,
) -> dict[str, Any]:
    df = get_table(table_name).copy()
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found in columns: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = np.isfinite(df[x_col].to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
        df[y_col].to_numpy(dtype=float, na_value=np.nan)
    )
    plot_df = df.loc[mask].copy()
    if log10:
        plot_df = plot_df[(plot_df[x_col] > 0) & (plot_df[y_col] > 0)].copy()
        plot_df[x_col] = np.log10(plot_df[x_col])
        plot_df[y_col] = np.log10(plot_df[y_col])
    if plot_df.empty:
        raise ValueError("no finite rows available for scatter plot")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(float(width), float(height)))
    if group_col and group_col in plot_df.columns:
        groups = [g for g in pd.unique(plot_df[group_col])]
        for i, group in enumerate(groups):
            part = plot_df[plot_df[group_col] == group]
            ax.scatter(
                part[x_col],
                part[y_col],
                s=12,
                alpha=0.65,
                color=_PALETTE[i % len(_PALETTE)],
                edgecolor="none",
                label=str(group),
            )
        if len(groups) > 1:
            ax.legend(frameon=False)
    else:
        groups = []
        ax.scatter(plot_df[x_col], plot_df[y_col], s=12, alpha=0.65, color=_PALETTE[0], edgecolor="none")

    slope = np.nan
    intercept = np.nan
    corr_p_value = np.nan
    if len(plot_df) >= 2:
        try:
            from scipy import stats as scipy_stats

            corr = scipy_stats.pearsonr(plot_df[x_col], plot_df[y_col])
            r = float(corr.statistic)
            corr_p_value = float(corr.pvalue)
        except Exception:
            r = float(np.corrcoef(plot_df[x_col], plot_df[y_col])[0, 1])
        ax.text(
            0.04,
            0.96,
            f"r = {r:.3g}" + (f"\n{_format_p_value(corr_p_value)}" if np.isfinite(corr_p_value) else ""),
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        if fit_line and np.isfinite(r):
            x = plot_df[x_col].to_numpy(dtype=float)
            y = plot_df[y_col].to_numpy(dtype=float)
            if np.nanmax(x) > np.nanmin(x):
                slope, intercept = np.polyfit(x, y, deg=1)
                xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                ax.plot(
                    xx,
                    slope * xx + intercept,
                    color="#111111",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=2,
                )
    else:
        r = np.nan
    ax.set_xlabel(xlabel or (f"log10 {x_col}" if log10 else x_col))
    ax.set_ylabel(ylabel or (f"log10 {y_col}" if log10 else y_col))
    if title:
        ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout()

    out = _figure_path(f"{table_name}__{x_col}__{y_col}__scatter", output_path, format)
    path = _save_figure(
        fig,
        out,
        dpi=dpi,
        metadata={
            "tool": "plot_scatter",
            "table_name": table_name,
            "x_col": x_col,
            "y_col": y_col,
            "log10": log10,
            "fit_line": fit_line,
            "pearson_r": r,
            "pearson_p_value": corr_p_value,
        },
    )
    plt.close(fig)

    return {
        "path": path,
        "format": format,
        "table_name": table_name,
        "x_col": x_col,
        "y_col": y_col,
        "n_points": int(len(plot_df)),
        "pearson_r": r,
        "pearson_p_value": corr_p_value,
        "fit_slope": float(slope) if np.isfinite(slope) else np.nan,
        "fit_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
        "groups": [str(g) for g in groups],
    }
