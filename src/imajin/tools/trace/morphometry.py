from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin import session as state
from imajin.tools._trace_store import _entry
from imajin.tools._trace_tables import _branch_summary, _node_table, _put_table
from imajin.tools.registry import tool


@tool(
    description="Compute a Sholl-style intersection profile around the soma or skeleton "
    "centroid. Stores a table with radius_um and intersections.",
    phase="6B",
    subagent="neural_tracer",
)
def compute_sholl_analysis(
    skeleton_id: str,
    center: str = "soma",
    radius_step_um: float = 5.0,
    max_radius_um: float | None = None,
) -> dict[str, Any]:
    if radius_step_um <= 0:
        raise ValueError("radius_step_um must be positive")
    entry = _entry(skeleton_id)
    coords = np.asarray(entry.skel.coordinates, dtype=float) * np.asarray(entry.record.spacing)
    if coords.size == 0:
        raise ValueError("skeleton has no coordinates")
    if center == "soma":
        if entry.record.soma is None:
            center_point = coords.mean(axis=0)
            center_used = "centroid"
        else:
            center_point = np.asarray(entry.record.soma, dtype=float)
            center_used = "soma"
    elif center == "centroid":
        center_point = coords.mean(axis=0)
        center_used = "centroid"
    else:
        parts = [float(v.strip()) for v in center.split(",")]
        if len(parts) != coords.shape[1]:
            raise ValueError(f"center must have {coords.shape[1]} comma-separated values")
        center_point = np.asarray(parts, dtype=float)
        center_used = "explicit"

    distances = np.linalg.norm(coords - center_point, axis=1)
    max_radius = float(max_radius_um) if max_radius_um is not None else float(distances.max())
    radii = np.arange(float(radius_step_um), max_radius + 1e-9, float(radius_step_um))
    graph = entry.skel.graph.tocoo()
    edge_pairs = [(int(s), int(d)) for s, d in zip(graph.row, graph.col, strict=False) if int(s) < int(d)]
    rows = []
    for radius in radii:
        count = 0
        for src, dst in edge_pairs:
            d0 = distances[src] - radius
            d1 = distances[dst] - radius
            if d0 == 0 or d1 == 0 or (d0 < 0 < d1) or (d1 < 0 < d0):
                count += 1
        rows.append({"radius_um": float(radius), "intersections": int(count)})
    df = pd.DataFrame(rows)
    table_name = _put_table(
        f"{skeleton_id}_sholl",
        df,
        spec={
            "op": "compute_sholl_analysis",
            "skeleton_id": skeleton_id,
            "center": center_used,
            "radius_step_um": radius_step_um,
        },
    )
    entry.record.table_names["sholl"] = table_name
    if df.empty:
        peak_count = 0
        peak_radius = 0.0
        auc = 0.0
    else:
        peak_idx = int(df["intersections"].idxmax())
        peak_count = int(df.loc[peak_idx, "intersections"])
        peak_radius = float(df.loc[peak_idx, "radius_um"])
        auc = float(np.trapezoid(df["intersections"], df["radius_um"])) if len(df) > 1 else 0.0
    return {
        "skeleton_id": skeleton_id,
        "table_name": table_name,
        "center": center_used,
        "n_radii": int(len(df)),
        "peak_intersections": peak_count,
        "peak_radius_um": peak_radius,
        "area_under_curve": auc,
    }


@tool(
    description="Compute aggregate neural morphology descriptors: total length, branch "
    "counts, endpoints, junctions, connected components, bounding box, and occupancy.",
    phase="6B",
    subagent="neural_tracer",
)
def compute_morphology_descriptors(skeleton_id: str) -> dict[str, Any]:
    entry = _entry(skeleton_id)
    df = _branch_summary(entry.skel, entry.record.spacing)
    nodes = _node_table(entry.skel, entry.record.spacing)
    length_col = "branch_length_um" if "branch_length_um" in df.columns else "branch_length"
    lengths = df[length_col] if length_col in df.columns else pd.Series(dtype=float)
    coords = np.asarray(entry.skel.coordinates, dtype=float) * np.asarray(entry.record.spacing)
    bbox = np.ptp(coords, axis=0) if len(coords) else np.zeros(len(entry.record.spacing))
    types = df.get("branch_type_code", pd.Series(dtype=int))
    result = {
        "skeleton_id": skeleton_id,
        "total_length": float(lengths.sum()) if len(lengths) else 0.0,
        "length_unit": "um" if length_col.endswith("_um") else "pixels",
        "mean_branch_length": float(lengths.mean()) if len(lengths) else 0.0,
        "median_branch_length": float(lengths.median()) if len(lengths) else 0.0,
        "n_branches": int(len(df)),
        "n_endpoints": int((nodes["degree"] == 1).sum()) if "degree" in nodes else 0,
        "n_junctions": int((nodes["degree"] > 2).sum()) if "degree" in nodes else 0,
        "n_components": int(entry.record.n_components),
        "n_terminal_branches": int(((types == 0) | (types == 1)).sum()) if len(types) else 0,
        "n_internal_branches": int((types == 2).sum()) if len(types) else 0,
        "bbox_scaled": tuple(float(v) for v in bbox),
        "skeleton_voxels": int(np.count_nonzero(entry.skeleton_image)),
        "skeleton_volume_occupancy": float(
            np.count_nonzero(entry.skeleton_image) / entry.skeleton_image.size
        ),
        "note": "Local morphology descriptors only. Connectome/NBLAST matching requires a backend plugin.",
    }
    state.put_qc_record(
        skeleton_id,
        status="pass",
        warnings=[],
        metrics={"kind": "neural_morphology", **result},
    )
    return result


