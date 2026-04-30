from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.state import get_layer, get_viewer, put_table
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


_BRANCH_TYPES = {
    0: "endpoint-endpoint",
    1: "junction-endpoint",
    2: "junction-junction",
    3: "isolated-cycle",
}


@dataclass
class _SkeletonRef:
    """Lightweight handle returned to the LLM as a JSON-friendly id; the actual
    skan.Skeleton object lives in _SKELETON_REGISTRY keyed by id."""

    id: str
    source_layer: str
    n_paths: int
    shape: tuple[int, ...]


_SKELETON_REGISTRY: dict[str, Any] = {}


def get_skeleton(skel_id: str):
    if skel_id not in _SKELETON_REGISTRY:
        raise KeyError(f"skeleton {skel_id!r} not found. Available: {list(_SKELETON_REGISTRY)}")
    return _SKELETON_REGISTRY[skel_id]


def reset_skeletons() -> None:
    _SKELETON_REGISTRY.clear()


@tool(
    description="Skeletonize a Labels or binary Image layer into a 1-pixel-wide "
    "centerline graph (skan.Skeleton). Works on 2D or 3D. Returns a skeleton id used "
    "by extract_branch_metrics and morphology tools. Adds a binary skeleton image "
    "layer for visual inspection.",
    phase="6",
    subagent="neural_tracer",
)
def skeletonize(layer: str, min_branch_length: float = 0.0) -> dict[str, Any]:
    from skan import Skeleton
    from skimage.morphology import skeletonize as sk_skeletonize

    L = get_layer(layer)
    data = _materialize(L.data)
    binary = (data > 0).astype(bool)
    if binary.ndim not in (2, 3):
        raise ValueError(
            f"skeletonize expects 2D or 3D layer; got shape {binary.shape}"
        )

    skel_image = sk_skeletonize(binary)
    if not skel_image.any():
        raise ValueError("skeleton is empty — input mask may be too small or disconnected")

    spacing = tuple(float(s) for s in L.scale) if hasattr(L, "scale") else 1.0
    skel = Skeleton(skel_image, spacing=spacing, keep_images=True)

    skel_id = f"skel_{len(_SKELETON_REGISTRY)}_{L.name}"
    _SKELETON_REGISTRY[skel_id] = skel

    viewer = get_viewer()
    viewer.add_image(
        skel_image.astype(np.uint8),
        name=f"{L.name}_skeleton",
        scale=tuple(float(s) for s in L.scale),
        metadata={"source_layer": L.name, "op": "skeletonize", "skeleton_id": skel_id},
        colormap="red",
        blending="additive",
    )

    return {
        "skeleton_id": skel_id,
        "source_layer": L.name,
        "n_paths": int(skel.n_paths),
        "shape": tuple(int(s) for s in skel_image.shape),
    }


@tool(
    description="Extract per-branch metrics from a skeleton: branch length (in scaled "
    "units, e.g. µm), branch type (endpoint-endpoint, junction-endpoint, "
    "junction-junction, isolated-cycle), euclidean distance between endpoints, "
    "and tortuosity (path/euclidean). Returns a table name.",
    phase="6",
    subagent="neural_tracer",
)
def extract_branch_metrics(skeleton_id: str) -> dict[str, Any]:
    from skan import summarize

    skel = get_skeleton(skeleton_id)
    df = summarize(skel, separator="-")

    rename = {
        "branch-distance": "branch_length",
        "branch-type": "branch_type_code",
        "euclidean-distance": "euclidean_distance",
        "skeleton-id": "skeleton_component",
    }
    for old, new in rename.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    if "branch_type_code" in df.columns:
        df["branch_type"] = df["branch_type_code"].map(_BRANCH_TYPES).fillna("unknown")
    if "branch_length" in df.columns and "euclidean_distance" in df.columns:
        ed = df["euclidean_distance"].replace(0, np.nan)
        df["tortuosity"] = df["branch_length"] / ed

    table_name = put_table(
        f"{skeleton_id}_branches",
        df,
        spec={"op": "extract_branch_metrics", "skeleton_id": skeleton_id},
    )

    counts: dict[str, int] = {}
    if "branch_type" in df.columns:
        counts = {k: int(v) for k, v in df["branch_type"].value_counts().to_dict().items()}

    return {
        "table_name": table_name,
        "n_branches": int(len(df)),
        "branch_type_counts": counts,
    }


@tool(
    description="Compute a morphology descriptor vector for a skeleton (NBLAST-compatible "
    "stub). Currently returns simple aggregate features (total length, mean branch "
    "length, n_endpoints, n_junctions). Real NBLAST integration via navis is deferred.",
    phase="6",
    subagent="neural_tracer",
)
def compute_morphology_descriptors(skeleton_id: str) -> dict[str, Any]:
    from skan import summarize

    skel = get_skeleton(skeleton_id)
    df = summarize(skel, separator="-")

    total_length = float(df.get("branch-distance", df.get("branch_length", pd.Series(dtype=float))).sum())
    mean_length = float(
        df.get("branch-distance", df.get("branch_length", pd.Series(dtype=float))).mean() or 0.0
    )
    types = df.get("branch-type", df.get("branch_type_code", pd.Series(dtype=int)))
    n_endpoint_endpoint = int((types == 0).sum()) if len(types) else 0
    n_junction_endpoint = int((types == 1).sum()) if len(types) else 0
    n_junction_junction = int((types == 2).sum()) if len(types) else 0

    return {
        "skeleton_id": skeleton_id,
        "total_length": total_length,
        "mean_branch_length": mean_length,
        "n_branches": int(len(df)),
        "n_terminal_branches": n_endpoint_endpoint + n_junction_endpoint,
        "n_internal_branches": n_junction_junction,
        "note": "Stub descriptors. Real NBLAST vectors require navis integration.",
    }


@tool(
    description="Query a connectome database for nearest reference neurons by morphology. "
    "STUB — returns 'not implemented' until a target organism / database backend is "
    "wired up (planned: FlyWire, neuPrint, MICrONS via navis).",
    phase="6",
    subagent="neural_tracer",
)
def query_connectome(
    skeleton_id: str,
    db: str = "neuprint",
    k: int = 10,
) -> dict[str, Any]:
    if db not in {"flywire", "neuprint", "microns", "allen"}:
        raise ValueError(f"unknown db {db!r}; expected flywire|neuprint|microns|allen")
    return {
        "skeleton_id": skeleton_id,
        "db": db,
        "k": k,
        "matches": [],
        "status": "not_implemented",
        "note": "Connectome backend deferred. Will use navis-neuprint / fafbseg / CAVE when target dataset is decided.",
    }


@tool(
    description="Classify a skeleton's neuron type by comparison to a reference set. "
    "STUB — returns 'not implemented' until a reference morphology DB or a learned "
    "classifier is added.",
    phase="6",
    subagent="neural_tracer",
)
def classify_neuron_type(
    skeleton_id: str,
    reference: str = "default",
) -> dict[str, Any]:
    return {
        "skeleton_id": skeleton_id,
        "reference": reference,
        "predicted_type": None,
        "confidence": None,
        "status": "not_implemented",
        "note": "Classification deferred. NBLAST-vs-reference or learned classifier coming with connectome integration.",
    }
