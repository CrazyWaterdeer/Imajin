from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any

from imajin.paths import normalize_user_path
from imajin.session import (
    get_table_entry,
    list_channel_annotations,
    list_files,
    list_qc_records,
    list_recipes,
    list_runs,
    list_samples,
    list_tables,
)
from imajin.tools.registry import tool


_TOOL_PHRASES: dict[str, str] = {
    "load_file": "loaded raw imaging data",
    "rolling_ball_background": (
        "background was subtracted using a rolling-ball algorithm "
        "(scikit-image, radius={radius})"
    ),
    "auto_contrast": (
        "intensities were rescaled to the {percentiles} percentile range "
        "(scikit-image)"
    ),
    "gaussian_denoise": (
        "Gaussian denoising was applied (sigma={sigma}, scikit-image)"
    ),
    "extract_timepoint": (
        "a reference timepoint (t={timepoint}) was extracted for ROI definition"
    ),
    "cellpose_sam": (
        "cells were segmented with Cellpose-SAM ({model_str}{do_3d_str}) on "
        "channel {channel}"
    ),
    "segment_intensity_regions": (
        "reporter-positive ROIs/regions were segmented by intensity thresholding "
        "({threshold_method}) on channel {channel}"
    ),
    "segment_target_objects": (
        "target-positive objects/ROIs were segmented after local background "
        "correction ({background_method}, radius={background_radius}px) on "
        "channel {channel}"
    ),
    "measure_intensity": (
        "per-object intensity features ({properties}) were extracted with "
        "scikit-image regionprops_table on channel(s) {channels}"
    ),
    "measure_intensity_over_time": (
        "ROI intensity time courses were extracted with scikit-image "
        "regionprops_table from {image_layer}"
    ),
    "measure_projected_intensity": (
        "ROI intensity was measured from a {projection} projection of {image_layer}"
    ),
    "manders_coefficients": (
        "Manders M1/M2 colocalization coefficients were computed between "
        "{image_a} and {image_b}"
    ),
    "pearson_correlation": (
        "Pearson correlation was computed between {image_a} and {image_b}"
    ),
    "max_projection": "maximum-intensity projection was generated along the {axis} axis",
    "average_projection": "average-intensity projection was generated along the {axis} axis",
    "export_channel_composite_png": (
        "a merged channel PNG was exported with {projection} projection and scale bar"
    ),
    "orthogonal_views": "orthogonal XZ and YZ projection views were generated",
    "enhance_neural_processes": (
        "neural process signal was enhanced ({method}) before tracing"
    ),
    "segment_neural_processes": (
        "neural processes were thresholded into a process mask ({threshold})"
    ),
    "skeletonize": "the segmentation was skeletonized (skan / scikit-image)",
    "extract_branch_metrics": (
        "per-branch morphology metrics (length, branch type, tortuosity) were "
        "extracted with skan"
    ),
    "prune_skeleton": "short skeleton branches were pruned below {min_branch_length_um}",
    "compute_sholl_analysis": "Sholl-style intersections were computed from the skeleton",
    "export_neural_trace": "neural trace data were exported as {format}",
    "track_cells": "cells were tracked across the time series with btrack",
    "describe_table": (
        "descriptive statistics were computed for {value_col}, including mean, "
        "median, dispersion, percentiles, and IQR outlier counts"
    ),
    "compare_groups": (
        "group-level statistical comparison was performed for {value_col} using "
        "{test}"
    ),
    "normalize_timecourse": (
        "time-course traces for {value_col} were normalized using {method}"
    ),
    "extract_timecourse_features": (
        "per-trace response features were extracted from {value_col}"
    ),
    "plot_group_distribution": (
        "a publication-style group distribution figure was exported for {value_col}"
    ),
    "plot_timecourse": "a publication-style time-course figure was exported for {value_col}",
    "plot_scatter": "a publication-style scatter figure was exported for {x_col} versus {y_col}",
    "analyze_target_cells": (
        "target-positive objects/ROIs were segmented from the user-confirmed "
        "target channel ({channel}) "
        "with {segmentation_method} ({do_3d_str}), and per-object intensity and size were "
        "measured on the same target channel"
    ),
}


def _format_phrase(tool_name: str, inputs: dict[str, Any]) -> str | None:
    template = _TOOL_PHRASES.get(tool_name)
    if template is None:
        return None
    args: dict[str, Any] = {}
    args["radius"] = inputs.get("radius", "?")
    args["sigma"] = inputs.get("sigma", "?")
    args["channel"] = (
        inputs.get("channel")
        or inputs.get("image_layer")
        or inputs.get("target")
        or inputs.get("target_channel")
        or "?"
    )
    args["channels"] = (
        inputs.get("channels")
        or inputs.get("image_layers")
        or inputs.get("image_layer")
        or "?"
    )
    args["properties"] = inputs.get("properties") or [
        "label",
        "area",
        "centroid",
        "mean_intensity",
        "max_intensity",
        "min_intensity",
    ]
    args["image_a"] = inputs.get("image_a", "?")
    args["image_b"] = inputs.get("image_b", "?")
    args["image_layer"] = inputs.get("image_layer", "?")
    args["axis"] = inputs.get("axis", "?")
    args["projection"] = inputs.get("projection", "?")
    args["method"] = inputs.get("method", "?")
    args["threshold"] = inputs.get("threshold", "?")
    args["threshold_method"] = inputs.get("threshold_method", "?")
    args["segmentation_method"] = inputs.get("segmentation_method", "Cellpose-SAM")
    args["background_method"] = inputs.get("background_method", "?")
    args["background_radius"] = inputs.get("background_radius", "?")
    args["format"] = inputs.get("format", "?")
    args["min_branch_length_um"] = inputs.get("min_branch_length_um", "?")
    args["value_col"] = inputs.get("value_col", "?")
    args["x_col"] = inputs.get("x_col", "?")
    args["y_col"] = inputs.get("y_col", "?")
    args["test"] = inputs.get("test", "auto")
    args["timepoint"] = inputs.get("t", inputs.get("timepoint", "?"))
    pct = inputs.get("percentiles")
    if pct is None:
        pct = (inputs.get("low_pct", 1.0), inputs.get("high_pct", 99.0))
    args["percentiles"] = f"{pct}"
    args["model_str"] = f"model={inputs.get('model', 'cpsam')!r}"
    args["do_3d_str"] = ", 3D" if inputs.get("do_3D") else ", 2D"
    try:
        return template.format(**args)
    except KeyError:
        return template


def _select_pipeline_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop housekeeping calls (list_layers, screenshot, export_*, refresh_measurement,
    summarize_table, filter_table) and any failures. Keep one entry per (tool, inputs)
    to avoid duplicating retried steps."""
    skip = {
        "list_layers",
        "get_layer_summary",
        "screenshot",
        "export_table",
        "save_labels",
        "save_result_bundle",
        "export_script",
        "refresh_measurement",
        "summarize_table",
        "filter_table",
        "set_view",
        "set_colormap",
        "compute_segmentation_qc",
        "compute_measurement_qc",
        "compute_timecourse_qc",
        "create_label_outline",
        "jump_to_object",
        "mark_qc_status",
        "annotate_sample",
        "list_sample_annotations",
        "annotate_channel",
        "list_channel_annotations",
        "resolve_channel",
        "consult_neural_tracer",
        "consult_methods_writer",
        "generate_methods",
        "generate_report",
    }
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in records:
        if not r.get("ok", True):
            continue
        if r["tool"] in skip:
            continue
        key = (r["tool"], repr(sorted((r.get("inputs") or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _render_methods_markdown(records: list[dict[str, Any]]) -> str:
    pipeline = _select_pipeline_records(records)
    if not pipeline:
        return (
            "## Methods\n\n"
            "No analytical operations were recorded for this session.\n"
        )

    sentences: list[str] = []
    for r in pipeline:
        phrase = _format_phrase(r["tool"], r.get("inputs") or {})
        if phrase:
            sentences.append(phrase)

    counts = Counter(r["tool"] for r in pipeline)
    duration = sum(float(r.get("duration_s", 0.0)) for r in pipeline)

    body = "; ".join(sentences) if sentences else "various analyses were performed"
    body = body[0].upper() + body[1:] if body else body

    methods = (
        "## Methods\n\n"
        f"Confocal image analysis was performed in napari ≥ 0.7 with the imajin "
        f"toolkit. {body}. All operations were executed against the original raw data "
        f"(no destructive overwrites), with a complete provenance log retained for "
        f"reproducibility.\n\n"
        f"_Provenance: {len(pipeline)} analysis operations across "
        f"{len(counts)} distinct tools, total compute time "
        f"{duration:.1f} s._\n"
    )
    return methods


def _render_samples_markdown(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return ""
    lines = ["## Sample Groups", ""]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        raw = sample.get("group")
        key = str(raw).strip() if raw not in (None, "") else "unassigned"
        grouped.setdefault(key, []).append(sample)
    for group, entries in sorted(grouped.items()):
        names = ", ".join(str(e.get("sample_name", "?")) for e in entries)
        lines.append(f"- **{group}**: {names}")
    lines.append("")
    return "\n".join(lines)


def _render_channels_markdown(channels: list[dict[str, Any]]) -> str:
    if not channels:
        return ""
    lines = ["## Channel Annotations", ""]
    for channel in channels:
        layer = channel.get("layer_name", "?")
        role = channel.get("role", "?")
        color = channel.get("color") or "unspecified"
        marker = channel.get("marker") or "unspecified"
        target = channel.get("biological_target")
        suffix = f", target={target}" if target else ""
        lines.append(f"- **{layer}**: {role}, {color}, marker={marker}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _render_qc_markdown(qc_records: list[dict[str, Any]]) -> str:
    if not qc_records:
        return ""
    lines = [
        "## Quality Control",
        "",
        "| Source | Status | Warnings | Reviewed |",
        "|---|---|---|---|",
    ]
    for record in qc_records:
        warnings = "; ".join(str(w) for w in (record.get("warnings") or []))
        warnings = warnings.replace("|", "/") or "—"
        source = str(record.get("source") or "—").replace("|", "/")
        status = str(record.get("status") or "not_checked").replace("|", "/")
        reviewed = "yes" if record.get("reviewed_by_user") else "no"
        lines.append(f"| {source} | {status} | {warnings} | {reviewed} |")
    lines.append("")
    return "\n".join(lines)


def _render_neural_traces_markdown(
    traces: list[dict[str, Any]],
    qc_records: list[dict[str, Any]],
) -> str:
    if not traces:
        return ""
    qc_by_source = {str(r.get("source")): r for r in qc_records}
    lines = [
        "## Neural Morphology",
        "",
        "| Trace | Source | Status | Paths | Components | Total length | Tables |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for trace in traces:
        trace_id = str(trace.get("trace_id") or "—")
        qc = qc_by_source.get(trace_id, {})
        metrics = qc.get("metrics") or {}
        total_length = metrics.get("total_length", "—")
        if isinstance(total_length, float):
            total_length = f"{total_length:.3g}"
        source = str(trace.get("source_layer") or "—").replace("|", "/")
        status = str(trace.get("status") or "—").replace("|", "/")
        table_names = trace.get("table_names") or {}
        tables = ", ".join(str(v) for v in table_names.values()) or "—"
        lines.append(
            f"| {trace_id} | {source} | {status} | {trace.get('n_paths', 0)} "
            f"| {trace.get('n_components', 0)} | {total_length} | {tables} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_report_html(
    records: list[dict[str, Any]],
    methods_md: str,
    acquisition_md: str = "",
    samples_md: str = "",
    channels_md: str = "",
    stats_md: str = "",
    qc_md: str = "",
    neural_md: str = "",
) -> str:
    pipeline = _select_pipeline_records(records)
    rows = "".join(
        f"<tr><td>{escape(str(r['tool']))}</td><td><code>{escape(_short_inputs(r.get('inputs') or {}))}</code></td>"
        f"<td>{r.get('duration_s', 0):.2f}s</td></tr>"
        for r in pipeline
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>imajin session report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:780px;margin:2em auto;"
        "padding:0 1em;line-height:1.5}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:.4em .6em;text-align:left}"
        "code{font-size:.9em}</style></head><body>"
        f"<h1>imajin session report</h1>"
        f"<pre>{escape(methods_md)}</pre>"
        f"{'<pre>' + escape(acquisition_md) + '</pre>' if acquisition_md else ''}"
        f"{'<pre>' + escape(samples_md) + '</pre>' if samples_md else ''}"
        f"{'<pre>' + escape(channels_md) + '</pre>' if channels_md else ''}"
        f"{'<pre>' + escape(stats_md) + '</pre>' if stats_md else ''}"
        f"{'<pre>' + escape(neural_md) + '</pre>' if neural_md else ''}"
        f"{'<pre>' + escape(qc_md) + '</pre>' if qc_md else ''}"
        f"<h2>Operations</h2><table><tr><th>Tool</th><th>Inputs</th><th>Time</th></tr>"
        f"{rows}</table></body></html>"
    )


def _short_inputs(inputs: dict[str, Any]) -> str:
    pairs = []
    for k, v in inputs.items():
        s = repr(v)
        if len(s) > 60:
            s = s[:57] + "..."
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def _append_optional_markdown(base: str, *sections: str) -> str:
    out = base
    for section in sections:
        if section:
            out += "\n" + section
    return out


@tool(
    description="Render a deterministic Methods paragraph (markdown) from the session "
    "provenance log. Lists each analytical step with parameters, suitable for pasting "
    "into a paper draft. No LLM involved.",
    phase="7",
)
def generate_methods(session_id: str | None = None) -> dict[str, Any]:
    from imajin.agent import provenance

    records = provenance.read_session(session_id)
    text = _render_methods_markdown(records)
    return {
        "session_id": session_id or provenance.current_session_id(),
        "n_records": len(records),
        "markdown": text,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _format_voxel(voxel: Any) -> str | None:
    if not voxel:
        return None
    try:
        return " × ".join(f"{float(v):g}" for v in voxel) + " µm"
    except (TypeError, ValueError):
        return None


def _render_acquisition_markdown(files: list[dict[str, Any]]) -> str:
    """Surface per-file acquisition metadata (voxel size, channels, settings) for
    reproducibility. Renders nothing when no file carries metadata."""
    blocks: list[str] = []
    for record in files:
        md = record.get("metadata_summary") or {}
        if not md:
            continue
        name = str(record.get("original_name") or record.get("file_id") or "file")

        facts: list[str] = []
        if md.get("file_type"):
            facts.append(f"format {md['file_type']}")
        voxel = _format_voxel(md.get("voxel_size_um"))
        if voxel:
            facts.append(f"voxel size {voxel}")
        if md.get("axes") and md.get("shape") is not None:
            facts.append(f"axes {md['axes']}, shape {tuple(md['shape'])}")
        if md.get("dtype"):
            facts.append(f"dtype {md['dtype']}")
        if md.get("n_channels"):
            facts.append(f"{md['n_channels']} channel(s)")

        block = [f"### {name}", ""]
        block.append("- " + "; ".join(facts) if facts else "- (no acquisition metadata recorded)")

        channel_meta = md.get("channel_metadata") or []
        if channel_meta:
            block += [
                "",
                "| Channel | Color | Ex (nm) | Em (nm) | Bit depth |",
                "|---|---|---|---|---|",
            ]
            for ch in channel_meta:
                cells = [
                    str(ch.get("name") or ch.get("channel_name") or "—"),
                    str(ch.get("color") or "—"),
                    _format_metric(ch.get("excitation_wavelength_nm") or ch.get("excitation")),
                    _format_metric(ch.get("emission_wavelength_nm") or ch.get("emission")),
                    _format_metric(ch.get("bit_depth")),
                ]
                block.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
        block.append("")
        blocks.append("\n".join(block))

    if not blocks:
        return ""
    return "\n".join(["## Acquisition", ""] + blocks)


@tool(
    description="Write a full session report to disk (HTML by default, or .md). Embeds "
    "the deterministic Methods paragraph plus an operations table from the provenance "
    "log.",
    phase="7",
)
def generate_report(
    path: str,
    session_id: str | None = None,
    format: str = "html",
) -> dict[str, Any]:
    from imajin.agent import provenance
    from imajin.tools.trace import list_trace_records

    if format not in ("html", "md"):
        raise ValueError(f"format must be 'html' or 'md', got {format!r}")

    records = provenance.read_session(session_id)
    methods = _render_methods_markdown(records)
    from imajin.tools import stats as _stats

    stats_generated = _stats.ensure_default_statistics(save_csv=True)
    stats_md = _render_statistics_markdown()
    files = list_files()
    acquisition_md = _render_acquisition_markdown(files)
    samples = list_samples()
    samples_md = _render_samples_markdown(samples)
    channels = list_channel_annotations()
    channels_md = _render_channels_markdown(channels)
    qc_records = list_qc_records()
    qc_md = _render_qc_markdown(qc_records)
    neural_traces = list_trace_records()
    neural_md = _render_neural_traces_markdown(neural_traces, qc_records)
    out = normalize_user_path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "md":
        out.write_text(
            _append_optional_markdown(
                methods,
                acquisition_md,
                samples_md,
                channels_md,
                stats_md,
                neural_md,
                qc_md,
            ),
            encoding="utf-8",
        )
    else:
        out.write_text(
            _render_report_html(
                records,
                methods,
                acquisition_md,
                samples_md,
                channels_md,
                stats_md,
                qc_md,
                neural_md,
            ),
            encoding="utf-8",
        )

    return {
        "path": str(out),
        "format": format,
        "session_id": session_id or provenance.current_session_id(),
        "n_records": len(records),
        "n_files": len(files),
        "n_samples": len(samples),
        "n_channels": len(channels),
        "n_qc_records": len(qc_records),
        "n_neural_traces": len(neural_traces),
        "n_statistics_generated": len(stats_generated),
    }


def _render_overview(files, samples, recipes) -> str:
    n_groups = len({s.get("group") for s in samples if s.get("group")})
    return (
        "## Overview\n\n"
        f"- Files registered: {len(files)}\n"
        f"- Samples: {len(samples)}\n"
        f"- Groups: {n_groups}\n"
        f"- Recipes: {len(recipes)}\n"
    )


def _render_sample_table(samples) -> str:
    if not samples:
        return "## Sample Table\n\n_No samples registered._\n"
    lines = ["## Sample Table", "", "| Sample | Group | Files | Notes |", "|---|---|---|---|"]
    for s in samples:
        files = ", ".join(s.get("file_ids") or []) or ", ".join(s.get("files") or [])
        notes = (s.get("notes") or "").replace("|", "/")
        extra = "; ".join(f"{k}={v}" for k, v in (s.get("extra") or {}).items())
        if extra:
            notes = f"{notes} ({extra})" if notes else extra
        lines.append(
            f"| {s.get('sample_name', '?')} | {s.get('group') or '—'} "
            f"| {files or '—'} | {notes or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_recipes(recipes) -> str:
    if not recipes:
        return "## Analysis Recipe\n\n_No recipes defined._\n"
    parts = ["## Analysis Recipe", ""]
    for r in recipes:
        parts.append(f"### {r.get('name')}")
        parts.append(f"- target channel: `{r.get('target_channel') or 'unspecified'}`")
        if r.get("preprocessing"):
            parts.append(f"- preprocessing: `{r['preprocessing']}`")
        if r.get("segmentation"):
            parts.append(f"- segmentation: `{r['segmentation']}`")
        if r.get("measurement"):
            parts.append(f"- measurement: `{r['measurement']}`")
        if r.get("notes"):
            parts.append(f"- notes: {r['notes']}")
        parts.append("")
    return "\n".join(parts)


def _render_runs(runs) -> str:
    if not runs:
        return "## Results\n\n_No runs recorded._\n"
    lines = ["## Results", "", "| Sample | Status | Objects | Tables |", "|---|---|---|---|"]
    for r in runs:
        n_obj = (r.get("summary") or {}).get("n_objects") or "—"
        tables = ", ".join(r.get("table_names") or []) or "—"
        lines.append(
            f"| {r.get('sample_id', '?')} | {r.get('status', '?')} | {n_obj} | {tables} |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_table_from_df(df, *, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    cols = [str(c) for c in shown.columns]
    lines = [
        "| " + " | ".join(c.replace("|", "/") for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _statistics_tables() -> tuple[
    list[tuple[str, object, dict[str, object]]],
    list[tuple[str, object, dict[str, object]]],
    list[tuple[str, object, dict[str, object]]],
]:

    summary_tables: list[tuple[str, object, dict[str, object]]] = []
    comparison_tables: list[tuple[str, object, dict[str, object]]] = []
    other_tables: list[tuple[str, object, dict[str, object]]] = []
    for name in list_tables():
        try:
            entry = get_table_entry(name)
        except KeyError:
            continue
        spec = dict(entry.spec or {})
        tool_name = str(spec.get("tool") or "")
        if tool_name == "batch_auto_statistics_input":
            continue
        item = (name, entry.df, spec)
        if (
            tool_name == "describe_table"
            or name.startswith("stats_object__")
            or name.startswith("stats_sample__")
        ):
            summary_tables.append(item)
        elif tool_name == "compare_groups" or name.startswith("stats_compare__"):
            comparison_tables.append(item)
        elif tool_name == "summarize_experiment" or name.startswith("summary_"):
            other_tables.append(item)
    return summary_tables, comparison_tables, other_tables


def _append_statistics_table(
    parts: list[str],
    name: str,
    df,
    spec: dict[str, object],
) -> None:
    tool_name = spec.get("tool") or "statistics"
    value_col = spec.get("value_col") or spec.get("measurement")
    level = spec.get("level") or spec.get("data_level")
    title_bits = [str(tool_name)]
    if value_col:
        title_bits.append(str(value_col))
    if level:
        title_bits.append(str(level))
    parts.append(f"### {name}")
    parts.append("- " + "; ".join(title_bits))
    parts.append("")
    parts.append(_markdown_table_from_df(df))
    parts.append("")


def _append_statistics_section(
    parts: list[str],
    heading: str,
    tables: list[tuple[str, object, dict[str, object]]],
) -> None:
    if not tables:
        return
    parts.append(f"### {heading}")
    parts.append("")
    for name, df, spec in tables:
        _append_statistics_table(parts, name, df, spec)


def _render_statistics_markdown() -> str:
    summary_tables, comparison_tables, other_tables = _statistics_tables()
    if not summary_tables and not comparison_tables and not other_tables:
        return "## Statistics\n\n_No statistical analysis tables found._\n"

    parts = ["## Statistics", ""]
    _append_statistics_section(parts, "Measurement Summary", summary_tables)
    _append_statistics_section(parts, "Statistical Tests", comparison_tables)
    _append_statistics_section(parts, "Other Summaries", other_tables)
    return "\n".join(parts)


def _render_warnings(runs) -> str:
    failed = [r for r in runs if r.get("status") == "failed"]
    if not failed:
        return "## Warnings\n\n_No failures recorded._\n"
    lines = ["## Warnings", ""]
    for r in failed:
        lines.append(
            f"- **{r.get('sample_id')}** failed: {r.get('error') or 'unknown error'}"
        )
    lines.append("")
    return "\n".join(lines)


@tool(
    description="Write a Phase-3 experiment-level report to disk (markdown or HTML). "
    "Includes overview, sample table, analysis recipe, per-sample results, the "
    "deterministic Methods paragraph from session provenance, and a warnings "
    "section listing failed samples.",
    phase="3",
)
def generate_experiment_report(
    path: str,
    session_id: str | None = None,
    format: str = "md",
) -> dict[str, Any]:
    from imajin.agent import provenance
    from imajin.tools.trace import list_trace_records

    if format not in ("md", "html"):
        raise ValueError(f"format must be 'md' or 'html', got {format!r}")

    files = list_files()
    samples = list_samples()
    recipes = list_recipes()
    runs = list_runs()
    qc_records = list_qc_records()
    neural_traces = list_trace_records()

    records = provenance.read_session(session_id)
    methods_md = _render_methods_markdown(records)
    from imajin.tools import stats as _stats

    stats_generated = _stats.ensure_default_statistics(save_csv=True)

    body = "\n".join(
        [
            "# Experiment Report",
            "",
            _render_overview(files, samples, recipes),
            _render_sample_table(samples),
            _render_recipes(recipes),
            _render_runs(runs),
            _render_statistics_markdown(),
            methods_md,
            _render_neural_traces_markdown(neural_traces, qc_records),
            _render_qc_markdown(qc_records),
            _render_warnings(runs),
        ]
    )

    out = normalize_user_path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        out.write_text(body, encoding="utf-8")
    else:
        out.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>imajin experiment report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:780px;"
            "margin:2em auto;padding:0 1em;line-height:1.5}</style>"
            f"</head><body><pre>{escape(body)}</pre></body></html>",
            encoding="utf-8",
        )

    return {
        "path": str(out),
        "format": format,
        "n_samples": len(samples),
        "n_groups": len({s.get("group") for s in samples if s.get("group")}),
        "n_files": len(files),
        "n_runs": len(runs),
        "n_qc_records": len(qc_records),
        "n_neural_traces": len(neural_traces),
        "n_failed": sum(1 for r in runs if r.get("status") == "failed"),
        "n_statistics_generated": len(stats_generated),
        "session_id": session_id or provenance.current_session_id(),
    }
