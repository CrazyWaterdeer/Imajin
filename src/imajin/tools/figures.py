from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from imajin.session import get_table, put_table
from imajin.paths import normalize_user_path
from imajin.result_bundles import bundle_output_path, register_output
from imajin.results import slugify_result_name
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
    filename = f"{slugify_result_name(stem)}.{suffix}"
    return bundle_output_path("figures", filename)


def _save_figure(fig: Any, out: Path, *, dpi: int, metadata: dict[str, Any]) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", transparent=False)
    try:
        register_output("figure", out, metadata)
    except ValueError:
        # Explicit user-supplied path outside the active bundle: save but skip index.
        pass
    return str(out)


def _sample_or_object_values(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    sample_col: str,
    level: Literal["auto", "sample", "object"],
    sample_agg: Literal["mean", "median"],
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in columns: {list(df.columns)}")
    if level in {"auto", "sample"} and sample_col in df.columns:
        grouped = df.groupby([sample_col, group_col], dropna=False, sort=False)
        plot_df = (
            grouped[value_col]
            .agg(n_objects="count", mean="mean", median="median")
            .reset_index()
        )
        if weight_col is not None:
            from imajin.tools.stats import _weighted_mean

            weighted = (
                grouped.apply(
                    lambda g: _weighted_mean(g[value_col], g[weight_col]),
                    include_groups=False,
                )
                .rename("weighted")
                .reset_index()
            )
            plot_df = plot_df.merge(weighted, on=[sample_col, group_col], how="left")
            plot_df["plot_value"] = plot_df["weighted"]
        else:
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
    weight_col: str | None = None,
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
            weight_col=weight_col,
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


def _draw_paired_lines(
    ax: Any, positions: np.ndarray, plot_df: pd.DataFrame, *, group_col: str, sample_col: str, groups: list[Any]
) -> None:
    """Connect the same sample across two groups (within-subject / paired designs)."""
    if len(groups) != 2 or sample_col not in plot_df.columns:
        return
    piv = plot_df.pivot_table(index=sample_col, columns=group_col, values="plot_value", aggfunc="first")
    g0, g1 = groups
    if g0 not in piv.columns or g1 not in piv.columns:
        return
    x0, x1 = float(positions[0]), float(positions[1])
    for _, row in piv.iterrows():
        v0, v1 = row.get(g0), row.get(g1)
        if pd.notna(v0) and pd.notna(v1):
            ax.plot([x0, x1], [float(v0), float(v1)], color="#999999", linewidth=0.7, alpha=0.7, zorder=2)


def _annotate_posthoc(
    ax: Any, positions: np.ndarray, values: list[np.ndarray], *, groups: list[Any], posthoc_rows: list[dict[str, Any]]
) -> None:
    """Stacked significance brackets for significant post-hoc pairs (p_adjusted < 0.05)."""
    gpos = {str(g): float(positions[i]) for i, g in enumerate(groups)}
    finite = [v[np.isfinite(v)] for v in values if len(v)]
    if not finite:
        return
    allv = np.concatenate(finite)
    y_max = float(allv.max())
    y_range = (y_max - float(allv.min())) or max(abs(y_max), 1.0)
    sig = [
        r for r in posthoc_rows
        if float(r.get("p_adjusted", 1.0)) < 0.05
        and str(r["group_a"]) in gpos and str(r["group_b"]) in gpos
    ]
    sig.sort(key=lambda r: abs(gpos[str(r["group_a"])] - gpos[str(r["group_b"])]))
    step, h = 0.09 * y_range, 0.02 * y_range
    for lvl, r in enumerate(sig):
        x0, x1 = sorted((gpos[str(r["group_a"])], gpos[str(r["group_b"])]))
        y = y_max + step * (lvl + 1)
        ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="#333333", linewidth=0.8)
        ax.text((x0 + x1) / 2, y + h, _p_stars(float(r["p_adjusted"])) or "*", ha="center", va="bottom", fontsize=8)
    if sig:
        ax.set_ylim(top=y_max + step * (len(sig) + 1))


@tool(
    description="Export a publication-style group distribution figure. kind picks the mark: "
    "'box' (box+points, default), 'bar' (mean+SEM bar), 'violin', or 'dots' (all points + "
    "mean±SEM crossbar — best for small n). Defaults to sample-level values when sample_name "
    "is present. paired=True draws connecting lines between the same sample across two groups "
    "(inside/outside, before/after). For 3+ groups it draws multiplicity-corrected post-hoc "
    "significance brackets (Games-Howell/Dunn). Style via palette, ymin/ymax, log_y, "
    "zero_baseline, point_size, jitter, show_points. Saves SVG/PDF/PNG into figures/.",
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
    weight_col: str | None = "auto",
    kind: Literal["box", "bar", "violin", "dots"] = "box",
    paired: bool = False,
    show_posthoc: bool = True,
    palette: list[str] | None = None,
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
    ymin: float | None = None,
    ymax: float | None = None,
    log_y: bool = False,
    zero_baseline: bool = False,
    point_size: float | None = None,
    jitter: float = 0.12,
    show_points: bool = True,
    store_plot_data: bool = True,
) -> dict[str, Any]:
    df, _dropped = finite_numeric_frame(get_table(table_name), value_col)
    if df.empty:
        raise ValueError(f"table {table_name!r} has no finite values in {value_col!r}")
    from imajin.tools.stats import resolve_weight_col

    weight = resolve_weight_col(df, weight_col, value_col)
    plot_df, data_level = _sample_or_object_values(
        df,
        value_col,
        group_col=group_col,
        sample_col=sample_col,
        level=level,
        sample_agg=sample_agg,
        weight_col=weight,
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
        weight_col=weight,
    )

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(float(width), float(height)))
    positions = np.arange(1, len(groups) + 1, dtype=float)
    colors = list(palette) if palette else list(_PALETTE)
    psize = float(point_size) if point_size is not None else (22.0 if data_level == "sample" else 10.0)

    if kind == "box":
        box = ax.boxplot(
            values, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#333333", "linewidth": 0.9},
            capprops={"color": "#333333", "linewidth": 0.9},
            boxprops={"color": "#333333", "linewidth": 0.9},
        )
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.18)
    elif kind == "violin":
        parts = ax.violinplot(values, positions=positions, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.22)
            body.set_edgecolor("#333333")

    if paired:
        _draw_paired_lines(ax, positions, plot_df, group_col=group_col, sample_col=sample_col, groups=groups)

    rng = np.random.default_rng(12345)
    for i, arr in enumerate(values):
        if arr.size == 0:
            continue
        x = float(positions[i])
        color = colors[i % len(colors)]
        mean = float(np.mean(arr))
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        if kind == "bar":
            ax.bar(x, mean, width=0.62, color=color, alpha=0.35, edgecolor=color, linewidth=1.0, zorder=1)
        if show_points:
            jit = rng.uniform(-abs(jitter), abs(jitter), size=arr.size)
            ax.scatter(
                np.full(arr.size, x) + jit, arr, s=psize,
                alpha=0.9 if data_level == "sample" else 0.5, color=color,
                edgecolor="#222222" if kind == "box" else "white", linewidth=0.5, zorder=3,
            )
        if kind in ("bar", "dots"):
            ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor="#111111", elinewidth=1.3, capsize=4, zorder=4)
        if kind == "dots":
            ax.plot([x - 0.22, x + 0.22], [mean, mean], color="#111111", linewidth=2.2, solid_capstyle="round", zorder=4)
        if kind in ("box", "violin"):
            ax.errorbar(x, mean, yerr=1.96 * sem, fmt="D", color="#111111", markersize=3.5, linewidth=0.9, capsize=2, zorder=4)

    ax.set_xticks(positions)
    if show_n:
        ax.set_xticklabels(
            [f"{g}\nn={len(v)}" for g, v in zip(groups, values, strict=False)], rotation=0, ha="center",
        )
    else:
        ax.set_xticklabels([str(g) for g in groups], rotation=25, ha="right")
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)

    if len(groups) >= 3 and show_posthoc and stats_result and stats_result.get("posthoc"):
        _annotate_posthoc(ax, positions, values, groups=groups, posthoc_rows=stats_result["posthoc"])
    elif p_label:
        _annotate_p_value(ax, positions=positions, values=values, p_value=p_value, label=p_label)

    _style_axes(ax)
    if log_y:
        ax.set_yscale("log")
    elif zero_baseline:
        ax.set_ylim(bottom=0.0)
    if ymin is not None or ymax is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(bottom=ymin if ymin is not None else lo, top=ymax if ymax is not None else hi)
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


@tool(
    description="ΔF/F0 raster/heatmap for a long-format calcium table: one row per "
    "ROI (label), time on x, colour = ΔF/F0. Surveys many traces at once.",
    phase="6",
    worker=True,
)
def plot_dff_heatmap(
    table_name: str,
    value_col: str,
    output_path: str | None = None,
    format: Literal["svg", "pdf", "png"] = "png",
    time_col: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    df = get_table(table_name)
    if value_col not in df.columns:
        raise ValueError(f"column {value_col!r} not found in columns: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError("table has no 'label' column for the heatmap rows")
    tcol = time_col or infer_time_column(df)
    piv = (
        df.pivot_table(index="label", columns=tcol, values=value_col, aggfunc="mean")
        .sort_index()
    )
    if piv.empty:
        raise ValueError("no rows available for the ΔF/F0 heatmap")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, max(1.5, 0.3 * piv.shape[0])))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", interpolation="nearest",
                   cmap="magma")
    ax.set_xlabel(str(tcol))
    ax.set_ylabel("ROI (label)")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(i) for i in piv.index])
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()

    out = _figure_path(f"{table_name}__{value_col}__dff_heatmap", output_path, format)
    path = _save_figure(
        fig,
        out,
        dpi=200,
        metadata={"tool": "plot_dff_heatmap", "table_name": table_name,
                  "value_col": value_col},
    )
    plt.close(fig)
    return {"path": path, "format": format, "table_name": table_name,
            "value_col": value_col, "n_traces": int(piv.shape[0])}
