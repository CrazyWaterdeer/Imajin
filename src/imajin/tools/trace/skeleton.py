from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_layer
from imajin.tools._trace_image import _binary_from_layer_data, _materialize
from imajin.tools._trace_store import (
    _BRANCH_QC_STATUSES,
    _entry,
    _register_skeleton,
    get_trace_record,
)
from imajin.tools._trace_tables import _branch_summary, _put_table, _scale_tuple
from imajin.tools.napari_ops import add_image_from_worker, snapshot_layer
from imajin.tools.registry import tool


@tool(
    description="Skeletonize a binary/Labels neural process layer into a centerline graph. "
    "Continuous image layers are rejected unless threshold is provided. Adds a skeleton "
    "overlay layer and node/edge/component tables.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def skeletonize(
    layer: str,
    min_branch_length: float = 0.0,
    threshold: float | None = None,
) -> dict[str, Any]:
    from skan import Skeleton
    from skimage.morphology import skeletonize as sk_skeletonize

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    binary = _binary_from_layer_data(data, layer_name=layer, threshold=threshold)
    if binary.ndim not in (2, 3):
        raise ValueError(f"skeletonize expects 2D or 3D layer; got shape {binary.shape}")

    skel_image = sk_skeletonize(binary).astype(bool)
    if not skel_image.any():
        raise ValueError("skeleton is empty — input mask may be too small or disconnected")

    spacing = _scale_tuple(tuple(L.scale), skel_image.ndim)
    skel = Skeleton(skel_image, spacing=spacing, keep_images=True)
    removed_paths: list[int] = []
    if min_branch_length > 0 and skel.n_paths:
        lengths = np.asarray(skel.path_lengths(), dtype=float)
        removed_paths = [int(i) for i in np.where(lengths < float(min_branch_length))[0]]
        if removed_paths and len(removed_paths) < int(skel.n_paths):
            skel = skel.prune_paths(removed_paths)
            skel_image = np.asarray(skel.skeleton_image).astype(bool)
        elif len(removed_paths) >= int(skel.n_paths):
            removed_paths = []

    layer_obj = call_on_main(
        add_image_from_worker,
        skel_image.astype(np.uint8),
        name=f"{L.name}_skeleton",
        scale=tuple(L.scale),
        metadata={"source_layer": L.name, "op": "skeletonize", "pending_skeleton_id": True},
        colormap="red",
        blending="additive",
    )
    skel_id = _register_skeleton(
        skel=skel,
        skeleton_image=skel_image,
        source_layer=L.name,
        mask_layer=L.name,
        skeleton_layer=layer_obj.name,
        spacing=spacing,
        parameters={
            "min_branch_length": min_branch_length,
            "threshold": threshold,
            "removed_paths": removed_paths,
        },
    )
    try:
        layer_obj.metadata["skeleton_id"] = skel_id
    except Exception:
        pass
    record = get_trace_record(skel_id)
    return {
        "skeleton_id": skel_id,
        "source_layer": L.name,
        "skeleton_layer": layer_obj.name,
        "n_paths": int(skel.n_paths),
        "n_components": int(record.n_components),
        "shape": tuple(int(s) for s in skel_image.shape),
        "spacing": spacing,
        "removed_paths": removed_paths,
        "table_names": dict(record.table_names),
    }


@tool(
    description="Extract per-branch metrics from a skeleton: branch length in scaled "
    "units, branch type, euclidean distance, tortuosity, and component id. Returns a "
    "table name.",
    phase="6B",
    subagent="neural_tracer",
)
def extract_branch_metrics(skeleton_id: str) -> dict[str, Any]:
    entry = _entry(skeleton_id)
    df = _branch_summary(entry.skel, entry.record.spacing)

    table_name = _put_table(
        f"{skeleton_id}_branches",
        df,
        spec={"op": "extract_branch_metrics", "skeleton_id": skeleton_id},
    )
    entry.record.table_names["branches"] = table_name

    counts: dict[str, int] = {}
    if "branch_type" in df.columns:
        counts = {k: int(v) for k, v in df["branch_type"].value_counts().to_dict().items()}

    length_col = "branch_length_um" if "branch_length_um" in df.columns else "branch_length"
    return {
        "table_name": table_name,
        "n_branches": int(len(df)),
        "branch_type_counts": counts,
        "total_length": float(df[length_col].sum()) if length_col in df.columns else 0.0,
        "length_unit": "um" if length_col.endswith("_um") else "pixels",
    }


@tool(
    description="Prune branches shorter than a physical/scaled length threshold. Keeps "
    "the original skeleton intact and creates a new pruned skeleton layer and trace id.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def prune_skeleton(
    skeleton_id: str,
    min_branch_length_um: float,
    remove_isolated: bool = True,
) -> dict[str, Any]:
    entry = _entry(skeleton_id)
    skel = entry.skel
    branch_df = _branch_summary(skel, entry.record.spacing)
    length_col = "branch_length_um" if "branch_length_um" in branch_df.columns else "branch_length"
    to_remove = branch_df[branch_df[length_col] < float(min_branch_length_um)]
    if not remove_isolated and "branch_type_code" in to_remove.columns:
        to_remove = to_remove[to_remove["branch_type_code"] != 0]
    indices = [int(v) for v in to_remove["branch_id"].tolist()]
    if not indices:
        return {
            "skeleton_id": skeleton_id,
            "new_skeleton_id": skeleton_id,
            "n_removed": 0,
            "n_paths": int(skel.n_paths),
        }
    if len(indices) >= int(skel.n_paths):
        raise ValueError("pruning threshold would remove all branches")

    pruned = skel.prune_paths(indices)
    pruned_image = np.asarray(pruned.skeleton_image).astype(bool)
    layer_obj = call_on_main(
        add_image_from_worker,
        pruned_image.astype(np.uint8),
        name=f"{skeleton_id}_pruned",
        scale=entry.record.spacing,
        metadata={
            "source_skeleton_id": skeleton_id,
            "op": "prune_skeleton",
            "min_branch_length_um": min_branch_length_um,
            "removed_branch_ids": indices,
        },
        colormap="red",
        blending="additive",
    )
    new_id = _register_skeleton(
        skel=pruned,
        skeleton_image=pruned_image,
        source_layer=entry.record.source_layer,
        mask_layer=entry.record.mask_layer,
        skeleton_layer=layer_obj.name,
        spacing=entry.record.spacing,
        parameters={
            "parent_skeleton_id": skeleton_id,
            "min_branch_length_um": min_branch_length_um,
            "remove_isolated": remove_isolated,
            "removed_branch_ids": indices,
        },
        status="pruned",
        parent_trace_id=skeleton_id,
    )
    return {
        "skeleton_id": skeleton_id,
        "new_skeleton_id": new_id,
        "n_removed": len(indices),
        "removed_branch_ids": indices,
        "n_paths_before": int(skel.n_paths),
        "n_paths_after": int(pruned.n_paths),
    }


@tool(
    description="Mark skeleton branches as accepted, rejected, or not_checked for manual "
    "review. Existing geometry is preserved; the review state is stored on the trace.",
    phase="6B",
    subagent="neural_tracer",
)
def set_branch_qc(
    skeleton_id: str,
    branch_ids: list[int],
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in _BRANCH_QC_STATUSES:
        raise ValueError("status must be accepted, rejected, or not_checked")
    entry = _entry(skeleton_id)
    max_id = int(entry.skel.n_paths) - 1
    bad = [bid for bid in branch_ids if int(bid) < 0 or int(bid) > max_id]
    if bad:
        raise ValueError(f"branch ids out of range for {skeleton_id}: {bad}")
    for bid in branch_ids:
        key = int(bid)
        entry.qc.branch_statuses[key] = status
        if status == "rejected" and key not in entry.qc.rejected_branch_ids:
            entry.qc.rejected_branch_ids.append(key)
        if status != "rejected" and key in entry.qc.rejected_branch_ids:
            entry.qc.rejected_branch_ids.remove(key)
        if reason:
            entry.qc.branch_reasons[key] = reason
    entry.record.status = "reviewed"
    return {
        "skeleton_id": skeleton_id,
        "status": status,
        "branch_ids": [int(v) for v in branch_ids],
        "n_rejected": len(entry.qc.rejected_branch_ids),
        "reason": reason,
    }


@tool(
    description="Set an optional soma/reference point for a skeleton from a points layer "
    "or mask layer. Used by Sholl analysis and soma-relative metrics.",
    phase="6B",
    subagent="neural_tracer",
)
def set_soma_location(
    skeleton_id: str,
    point_layer: str | None = None,
    mask_layer: str | None = None,
) -> dict[str, Any]:
    from scipy import ndimage as ndi

    entry = _entry(skeleton_id)
    if point_layer is None and mask_layer is None:
        raise ValueError("provide point_layer or mask_layer")
    if point_layer is not None:
        L = get_layer(point_layer)
        data = _materialize(L.data)
        if data.size == 0:
            raise ValueError(f"point layer {point_layer!r} is empty")
        point = np.asarray(data, dtype=float).reshape(-1, data.shape[-1])[0]
    else:
        L = get_layer(str(mask_layer))
        data = _materialize(L.data) > 0
        if not data.any():
            raise ValueError(f"mask layer {mask_layer!r} is empty")
        point = np.asarray(ndi.center_of_mass(data), dtype=float)
        point = point * np.asarray(_scale_tuple(tuple(getattr(L, "scale", ())), data.ndim))
    entry.record.soma = tuple(float(v) for v in point)
    return {"skeleton_id": skeleton_id, "soma": entry.record.soma}


@tool(
    description="Assign a dominant anatomical region label to a skeleton by sampling a "
    "region Labels layer at skeleton node coordinates.",
    phase="6B",
    subagent="neural_tracer",
)
def assign_neural_region(skeleton_id: str, region_layer: str) -> dict[str, Any]:
    entry = _entry(skeleton_id)
    L = get_layer(region_layer)
    regions = _materialize(L.data)
    coords = np.asarray(entry.skel.coordinates, dtype=int)
    valid = np.ones(len(coords), dtype=bool)
    for axis in range(coords.shape[1]):
        valid &= (coords[:, axis] >= 0) & (coords[:, axis] < regions.shape[axis])
    sampled = regions[tuple(coords[valid].T)] if valid.any() else np.array([])
    sampled = sampled[sampled > 0]
    if sampled.size == 0:
        region: int | str | None = None
    else:
        values, counts = np.unique(sampled, return_counts=True)
        region = int(values[int(np.argmax(counts))])
    entry.record.region = region
    return {"skeleton_id": skeleton_id, "region_layer": region_layer, "region": region}


