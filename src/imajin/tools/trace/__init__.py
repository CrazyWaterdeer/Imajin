from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imajin.analysis.morphology_features import extract_feature_vector
from imajin.analysis.morphology_match import match_against_library
from imajin.analysis.morphology_reference import append_reference, load_reference_library
from imajin import session as state
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_layer
from imajin.paths import normalize_user_path
from imajin.tools._trace_export import _swc_coordinates, _write_swc
from imajin.tools._trace_image import (
    _binary_from_layer_data,
    _component_labels,
    _materialize,
    _normalize_image,
    _rolling_ball_subtract,
)
from imajin.tools._trace_store import (
    _BRANCH_QC_STATUSES,
    _SKELETON_REGISTRY,
    _SkeletonEntry,
    _TRACE_STATUSES,
    _entry,
    _register_skeleton,
    _store_graph_tables,
    NeuralTraceQC,
    NeuralTraceRecord,
    get_skeleton,
    get_trace_record,
    list_trace_records,
    reset_skeletons,
)
from imajin.tools._trace_tables import (
    _BRANCH_TYPES,
    _branch_summary,
    _component_table,
    _edge_table,
    _node_table,
    _normalize_branch_df,
    _put_table,
    _scale_is_physical,
    _scale_tuple,
)
from imajin.tools.napari_ops import (
    add_image_from_worker,
    add_labels_from_worker,
    snapshot_layer,
)
from imajin.tools.registry import tool

# Tool families live in submodules; import them for @tool registration and to
# re-export the public tool names on ``imajin.tools.trace``.
from imajin.tools.trace.enhance import (  # noqa: F401,E402
    enhance_neural_processes,
    segment_neural_processes,
)
from imajin.tools.trace.skeleton import (  # noqa: F401,E402
    assign_neural_region,
    extract_branch_metrics,
    prune_skeleton,
    set_branch_qc,
    set_soma_location,
    skeletonize,
)


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


@tool(
    description="Query an external connectome database (neuPrint/FlyWire, Drosophila) for "
    "reference neurons by morphology. Tier-2 backend — currently returns "
    "'not_implemented'. For local, offline morphology search use find_similar_neurons.",
    phase="6B",
    subagent="neural_tracer",
)
def query_connectome(
    skeleton_id: str,
    db: str = "neuprint",
    k: int = 10,
) -> dict[str, Any]:
    db = db.lower().strip()
    if db in {"microns", "allen"}:
        return {
            "skeleton_id": skeleton_id,
            "db": db,
            "matches": [],
            "status": "off_domain",
            "note": f"{db!r} is a mouse connectome; this app targets Drosophila. Not supported.",
        }
    if db not in {"flywire", "neuprint"}:
        raise ValueError(
            f"unknown db {db!r}; expected neuprint|flywire (microns/allen are mouse, off-domain)"
        )
    if db == "neuprint":
        from imajin.analysis.connectome_neuprint import query_neuprint

        # backend/token readiness is resolved without a skeleton lookup, so a bad id
        # returns a graceful status rather than KeyError (the live fetch — pending a
        # token + template registration — is not wired yet).
        result = query_neuprint(None, None, k=k)
        return {"skeleton_id": skeleton_id, "db": db, **result}

    # FlyWire is a later Tier-2 step (heavier CAVE auth); not wired yet.
    return {
        "skeleton_id": skeleton_id,
        "db": db,
        "k": k,
        "matches": [],
        "status": "not_implemented",
        "note": "FlyWire backend not wired yet (Tier 2, after neuPrint).",
    }


def _reference_library_path(reference: str) -> Path:
    """Resolve a reference-library path. 'default' → <results_root>/morphology_reference.csv."""
    if reference.strip() in ("", "default"):
        from imajin.results import results_root

        return results_root() / "morphology_reference.csv"
    return normalize_user_path(reference)


def _load_reference_or_none(reference: str):
    """Load the reference library, or None if it is missing/empty/malformed."""
    path = _reference_library_path(reference)
    if not path.exists():
        return None
    try:
        return load_reference_library(path)
    except (FileNotFoundError, ValueError):
        return None


def _add_persistence_features(feature_vector: dict[str, Any], entry: Any) -> None:
    """Enrich the feature vector with navis persistence features, when available.

    Persistence is rotation/translation-invariant but scale-sensitive, so the
    features are added to ``features`` but NOT to ``invariant_keys`` — the matcher
    uses them only when query and library share physical (micron) units. A no-op
    without the connectome extra or on a degenerate skeleton.
    """
    import tempfile

    from imajin.analysis.morphology_persistence import (
        persistence_available,
        persistence_features_from_swc,
    )

    if not persistence_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        swc = Path(tmp) / "skeleton.swc"
        _write_swc(entry, swc)
        features = persistence_features_from_swc(swc)
    if features:
        feature_vector["features"].update(features)


def _skeleton_feature_vector(skeleton_id: str) -> dict[str, Any]:
    """Descriptor → feature vector for one registered skeleton (incl. tortuosity)."""
    entry = _entry(skeleton_id)
    descriptors = compute_morphology_descriptors(skeleton_id)
    branch_df = _branch_summary(entry.skel, entry.record.spacing)
    fv = extract_feature_vector(descriptors, branch_df)
    _add_persistence_features(fv, entry)
    return fv


@tool(
    description="Classify a skeleton's neuron type by morphometric similarity to a "
    "labelled reference library (build one with add_reference_neuron). Local, "
    "offline, registration-free. Returns status 'no_reference' when no library is "
    "configured. (Spatial NBLAST / connectome lookup is a separate Tier-2 backend.)",
    phase="6B",
    subagent="neural_tracer",
)
def classify_neuron_type(
    skeleton_id: str,
    reference: str = "default",
) -> dict[str, Any]:
    # H3: resolve the reference library BEFORE touching the skeleton registry, so a
    # missing library returns a graceful status rather than KeyError on a bad id.
    library = _load_reference_or_none(reference)
    if library is None:
        return {
            "skeleton_id": skeleton_id,
            "reference": reference,
            "predicted_type": None,
            "confidence": None,
            "status": "no_reference",
            "note": (
                "No morphology reference library found. Build one by labelling your "
                "own traced neurons: add_reference_neuron(skeleton_id, label)."
            ),
        }

    fv = _skeleton_feature_vector(skeleton_id)
    res = match_against_library(fv, library, k=5)
    runner_up = res["ranked"][1]["label"] if len(res["ranked"]) > 1 else None

    # H2: distinct QC key — do NOT reuse the bare skeleton_id, which holds the
    # neural_morphology record written by compute_morphology_descriptors.
    state.put_qc_record(
        f"{skeleton_id}::classification",
        status="pass",
        metrics={
            "kind": "neural_classification",
            "predicted_type": res["predicted"],
            "confidence": res["confidence"],
            "invariant_only": res["invariant_only"],
        },
    )
    return {
        "skeleton_id": skeleton_id,
        "reference": reference,
        "predicted_type": res["predicted"],
        "confidence": res["confidence"],
        "runner_up": runner_up,
        "ranked": res["ranked"],
        "invariant_only": res["invariant_only"],
        "status": res["status"],
        "note": "Morphometric (feature-vector) match — registration-free, local.",
    }


@tool(
    description="Add the current skeleton to a labelled morphology reference library "
    "(CSV) so future neurons can be classified against it. Builds the library from "
    "your own traced + labelled neurons; fully local/offline.",
    phase="6B",
    subagent="neural_tracer",
)
def add_reference_neuron(
    skeleton_id: str,
    label: str,
    library_path: str = "default",
) -> dict[str, Any]:
    # adding a reference requires a real skeleton, so the lookup (KeyError on a bad
    # id) is the correct behaviour here
    fv = _skeleton_feature_vector(skeleton_id)
    path = _reference_library_path(library_path)
    library = append_reference(path, fv, label=label.strip(), name=skeleton_id)
    return {
        "skeleton_id": skeleton_id,
        "label": label.strip(),
        "library_path": str(path),
        "n_references": len(library),
        "units_physical": fv["units_physical"],
        "status": "ok",
    }


@tool(
    description="Find the k most morphometrically similar neurons in a labelled "
    "reference library (local, offline, registration-free). Returns status "
    "'no_reference' when no library is configured. (External connectome lookup is "
    "query_connectome, a separate Tier-2 backend.)",
    phase="6B",
    subagent="neural_tracer",
)
def find_similar_neurons(
    skeleton_id: str,
    reference: str = "default",
    k: int = 10,
) -> dict[str, Any]:
    # H3 ordering: reference first, skeleton lookup only when a library exists
    library = _load_reference_or_none(reference)
    if library is None:
        return {
            "skeleton_id": skeleton_id,
            "reference": reference,
            "matches": [],
            "status": "no_reference",
            "note": (
                "No morphology reference library found. Build one with "
                "add_reference_neuron(skeleton_id, label)."
            ),
        }

    fv = _skeleton_feature_vector(skeleton_id)
    res = match_against_library(fv, library, k=k)
    return {
        "skeleton_id": skeleton_id,
        "reference": reference,
        "matches": res["ranked"],
        "invariant_only": res["invariant_only"],
        "status": res["status"],
    }
