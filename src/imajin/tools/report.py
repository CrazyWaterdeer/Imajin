from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

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
    "measure_intensity": (
        "per-object intensity features ({properties}) were extracted with "
        "scikit-image regionprops_table on channel(s) {channels}"
    ),
    "measure_intensity_over_time": (
        "ROI intensity time courses were extracted with scikit-image "
        "regionprops_table from {image_layer}"
    ),
    "manders_coefficients": (
        "Manders M1/M2 colocalization coefficients were computed between "
        "{image_a} and {image_b}"
    ),
    "pearson_correlation": (
        "Pearson correlation was computed between {image_a} and {image_b}"
    ),
    "max_projection": "maximum-intensity projection was generated along the {axis} axis",
    "orthogonal_views": "orthogonal XZ and YZ projection views were generated",
    "skeletonize": "the segmentation was skeletonized (skan / scikit-image)",
    "extract_branch_metrics": (
        "per-branch morphology metrics (length, branch type, tortuosity) were "
        "extracted with skan"
    ),
    "track_cells": "cells were tracked across the time series with btrack",
    "analyze_target_cells": (
        "cells were segmented from the user-confirmed target channel ({channel}) "
        "with Cellpose-SAM ({do_3d_str}), and per-object intensity and size were "
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
        "export_script",
        "refresh_measurement",
        "summarize_table",
        "filter_table",
        "set_view",
        "set_colormap",
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


def _render_report_html(
    records: list[dict[str, Any]],
    methods_md: str,
    samples_md: str = "",
    channels_md: str = "",
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
        f"{'<pre>' + escape(samples_md) + '</pre>' if samples_md else ''}"
        f"{'<pre>' + escape(channels_md) + '</pre>' if channels_md else ''}"
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
    from imajin.agent.state import list_channel_annotations, list_samples

    if format not in ("html", "md"):
        raise ValueError(f"format must be 'html' or 'md', got {format!r}")

    records = provenance.read_session(session_id)
    methods = _render_methods_markdown(records)
    samples = list_samples()
    samples_md = _render_samples_markdown(samples)
    channels = list_channel_annotations()
    channels_md = _render_channels_markdown(channels)
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "md":
        extra = ""
        if samples_md:
            extra += "\n" + samples_md
        if channels_md:
            extra += "\n" + channels_md
        out.write_text(methods + extra, encoding="utf-8")
    else:
        out.write_text(
            _render_report_html(records, methods, samples_md, channels_md),
            encoding="utf-8",
        )

    return {
        "path": str(out),
        "format": format,
        "session_id": session_id or provenance.current_session_id(),
        "n_records": len(records),
        "n_samples": len(samples),
        "n_channels": len(channels),
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
    from imajin.agent.state import (
        list_files,
        list_recipes,
        list_runs,
        list_samples,
    )

    if format not in ("md", "html"):
        raise ValueError(f"format must be 'md' or 'html', got {format!r}")

    files = list_files()
    samples = list_samples()
    recipes = list_recipes()
    runs = list_runs()

    records = provenance.read_session(session_id)
    methods_md = _render_methods_markdown(records)

    body = "\n".join(
        [
            "# Experiment Report",
            "",
            _render_overview(files, samples, recipes),
            _render_sample_table(samples),
            _render_recipes(recipes),
            _render_runs(runs),
            methods_md,
            _render_warnings(runs),
        ]
    )

    out = Path(path).expanduser().resolve()
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
        "n_failed": sum(1 for r in runs if r.get("status") == "failed"),
        "session_id": session_id or provenance.current_session_id(),
    }
