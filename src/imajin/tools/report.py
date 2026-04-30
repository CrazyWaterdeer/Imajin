from __future__ import annotations

from collections import Counter
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
    "cellpose_sam": (
        "cells were segmented with Cellpose-SAM ({model_str}{do_3d_str}) on "
        "channel {channel}"
    ),
    "measure_intensity": (
        "per-object intensity features ({properties}) were extracted with "
        "scikit-image regionprops_table on channel(s) {channels}"
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
}


def _format_phrase(tool_name: str, inputs: dict[str, Any]) -> str | None:
    template = _TOOL_PHRASES.get(tool_name)
    if template is None:
        return None
    args: dict[str, Any] = {}
    args["radius"] = inputs.get("radius", "?")
    args["sigma"] = inputs.get("sigma", "?")
    args["channel"] = inputs.get("channel", "?")
    args["channels"] = inputs.get("channels", "?")
    args["properties"] = inputs.get("properties", "?")
    args["image_a"] = inputs.get("image_a", "?")
    args["image_b"] = inputs.get("image_b", "?")
    args["axis"] = inputs.get("axis", "?")
    pct = inputs.get("percentiles", "(1, 99)")
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


def _render_report_html(records: list[dict[str, Any]], methods_md: str) -> str:
    pipeline = _select_pipeline_records(records)
    rows = "".join(
        f"<tr><td>{r['tool']}</td><td><code>{_short_inputs(r.get('inputs') or {})}</code></td>"
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
        f"<pre>{methods_md}</pre>"
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

    if format not in ("html", "md"):
        raise ValueError(f"format must be 'html' or 'md', got {format!r}")

    records = provenance.read_session(session_id)
    methods = _render_methods_markdown(records)
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "md":
        out.write_text(methods, encoding="utf-8")
    else:
        out.write_text(_render_report_html(records, methods), encoding="utf-8")

    return {
        "path": str(out),
        "format": format,
        "session_id": session_id or provenance.current_session_id(),
        "n_records": len(records),
    }
