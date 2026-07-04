from __future__ import annotations

from typing import Any

import numpy as np

from imajin.paths import normalize_user_path
from imajin.tools._trace_export import _write_swc
from imajin.tools._trace_store import _entry
from imajin.tools._trace_tables import _branch_summary, _edge_table, _node_table
from imajin.tools.registry import tool


@tool(
    description="Export neural trace data. Formats: swc, csv (nodes/edges/branches), "
    "or tiff/tif skeleton image. SWC documents limitations when no soma/root is known.",
    phase="6B",
    subagent="neural_tracer",
)
def export_neural_trace(
    skeleton_id: str,
    output_path: str,
    format: str = "swc",
) -> dict[str, Any]:
    entry = _entry(skeleton_id)
    fmt = format.lower().strip()
    out = normalize_user_path(output_path).resolve()
    written: list[str] = []

    if fmt == "csv":
        out.mkdir(parents=True, exist_ok=True)
        nodes = _node_table(entry.skel, entry.record.spacing)
        edges = _edge_table(entry.skel, entry.record.spacing)
        branches = _branch_summary(entry.skel, entry.record.spacing)
        files = {
            "nodes": out / f"{skeleton_id}_nodes.csv",
            "edges": out / f"{skeleton_id}_edges.csv",
            "branches": out / f"{skeleton_id}_branches.csv",
        }
        nodes.to_csv(files["nodes"], index=False)
        edges.to_csv(files["edges"], index=False)
        branches.to_csv(files["branches"], index=False)
        written = [str(p) for p in files.values()]
    elif fmt in {"tif", "tiff"}:
        import tifffile

        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out, entry.skeleton_image.astype(np.uint8) * 255)
        written = [str(out)]
    elif fmt == "swc":
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_swc(entry, out)
        written = [str(out)]
    else:
        raise ValueError("format must be swc, csv, tif, or tiff")

    entry.record.status = "exported"
    return {
        "skeleton_id": skeleton_id,
        "format": fmt,
        "paths": written,
        "note": (
            "SWC export uses local skeleton topology. Soma/root assignment is approximate "
            "unless set_soma_location was called."
            if fmt == "swc"
            else None
        ),
    }


