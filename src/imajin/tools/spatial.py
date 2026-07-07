"""Object-to-object spatial relationships — the layer that turns two detections
into biology: how many spots per cell, how far each object is from a surface, and
how clustered a population is.

Objects may be a **Points** layer (from ``detect_spots``) or a **Labels** layer
(from segmentation). Distances are physical µm (voxel-scale aware). Every tool
emits a per-object table that flows into ``compare_groups`` / ``plot_*`` and
carries ids so aggregation stays honest.

Assumption: objects and the parent/reference layer share the same voxel grid (the
usual case — detected/segmented on the same image). Geometry follows the
coordinate contract in ``analysis/coords.py`` (data-coord layers, µm in tables).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis import coords
from imajin.analysis.arrays import materialize_array
from imajin.session import put_table
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


@dataclass
class _Objects:
    kind: str  # "points" | "labels"
    ndim: int
    ids: np.ndarray  # (N,) object ids: spot index or label value
    centroids: np.ndarray  # (N, ndim) index coordinates
    labels: np.ndarray | None  # label image when kind == "labels"
    spacing: tuple[float, ...]
    shape: tuple[int, ...] | None
    id_col: str  # "spot_id" or "label"


def _extract_objects(layer_name: str) -> _Objects:
    snap = call_on_main(snapshot_layer, layer_name)
    data = materialize_array(snap.data)
    if snap.kind == "points":
        c = np.asarray(data, dtype=float).reshape(-1, data.shape[-1]) if data.size else data.reshape(0, data.shape[-1])
        ndim = int(c.shape[1])
        return _Objects(
            "points", ndim, np.arange(len(c)), c, None,
            coords.layer_scale(snap, ndim), None, "spot_id",
        )
    lab = np.asarray(data)
    if lab.ndim not in (2, 3):
        raise ValueError(f"{layer_name!r} must be a 2D/3D Labels layer; got shape {lab.shape}")
    ndim = int(lab.ndim)
    spacing = coords.layer_scale(snap, ndim)
    present = np.unique(lab)
    present = present[present > 0]
    if len(present):
        from skimage.measure import regionprops_table

        rp = regionprops_table(lab.astype(int), properties=["label", "centroid"])
        ids = np.asarray(rp["label"], dtype=int)
        cent = np.column_stack([rp[f"centroid-{i}"] for i in range(ndim)])
    else:
        ids, cent = np.array([], dtype=int), np.empty((0, ndim))
    return _Objects("labels", ndim, ids, cent, lab.astype(np.int32), spacing, tuple(lab.shape), "label")


def _store(name: str, df: pd.DataFrame, spec: dict[str, Any]) -> str:
    return call_on_main(put_table, name, df, spec=spec)


@tool(
    description="Assign each object (a Points layer from detect_spots, or a Labels "
    "layer) to the parent Labels object that contains it — 'spots per cell'. Points "
    "use containment; label objects use maximum overlap, with an overlap fraction and "
    "an ambiguity flag when an object straddles two parents. Emits a per-object table "
    "(object_id, parent_id, overlap_fraction, assignment_ambiguous) and a per-parent "
    "summary (n_objects, density). Parent 0 means background / unassigned.",
    phase="4",
    worker=True,
)
def assign_objects_to_parents(
    objects_layer: str,
    parents_layer: str,
    table_name: str | None = None,
) -> dict[str, Any]:
    parents_snap = call_on_main(snapshot_layer, parents_layer)
    parents = materialize_array(parents_snap.data).astype(np.int32)
    if parents.ndim not in (2, 3):
        raise ValueError("parents_layer must be a 2D/3D Labels layer")
    spacing = coords.layer_scale(parents_snap, parents.ndim)
    obj = _extract_objects(objects_layer)

    object_ids: list[int] = []
    parent_ids: list[int] = []
    overlaps: list[float] = []
    ambiguous: list[bool] = []

    if obj.kind == "points":
        for oid, c in zip(obj.ids, obj.centroids):
            vox = tuple(int(round(v)) for v in c[: parents.ndim])
            in_bounds = all(0 <= vox[a] < parents.shape[a] for a in range(parents.ndim))
            pid = int(parents[vox]) if in_bounds else 0
            object_ids.append(int(oid))
            parent_ids.append(pid)
            overlaps.append(1.0 if pid > 0 else 0.0)
            ambiguous.append(False)
    else:
        if obj.shape != tuple(parents.shape):
            raise ValueError(
                f"objects {obj.shape} and parents {parents.shape} must share a grid"
            )
        for oid in obj.ids:
            omask = obj.labels == oid
            total = int(omask.sum())
            counts = np.bincount(parents[omask].ravel())
            nonzero = counts.copy()
            if nonzero.size:
                nonzero[0] = 0
            if nonzero.size and nonzero.max() > 0:
                pid = int(np.argmax(nonzero))
                frac = nonzero[pid] / total if total else 0.0
                others = [c for p, c in enumerate(nonzero) if p != pid and c > 0]
                amb = bool(others and (max(others) / total) > 0.2)
            else:
                pid, frac, amb = 0, 0.0, False
            object_ids.append(int(oid))
            parent_ids.append(pid)
            overlaps.append(float(frac))
            ambiguous.append(amb)

    df = pd.DataFrame(
        {
            "object_id": object_ids,
            "parent_id": parent_ids,
            "overlap_fraction": overlaps,
            "assignment_ambiguous": ambiguous,
        }
    )
    tname = table_name or f"{obj.id_col}_in_{parents_layer}"
    stored = _store(tname, df, {
        "op": "assign_objects_to_parents",
        "objects_layer": objects_layer,
        "parents_layer": parents_layer,
    })

    # Per-parent summary: object count + density over each parent's physical size.
    vox_um = float(np.prod(spacing))
    parent_sizes = np.bincount(parents.ravel())
    rows = []
    assigned = df[df["parent_id"] > 0]
    for pid, grp in assigned.groupby("parent_id"):
        size_vox = int(parent_sizes[pid]) if pid < len(parent_sizes) else 0
        vol = size_vox * vox_um
        rows.append({
            "parent_id": int(pid),
            "n_objects": int(len(grp)),
            "parent_size_um": float(vol),
            "density": float(len(grp) / vol) if vol else 0.0,
        })
    summary = pd.DataFrame(rows, columns=["parent_id", "n_objects", "parent_size_um", "density"])
    summary_name = _store(f"{tname}_by_parent", summary, {
        "op": "assign_objects_to_parents_summary", "objects_layer": objects_layer,
    })

    return {
        "table_name": stored,
        "summary_table": summary_name,
        "n_objects": int(len(df)),
        "n_assigned": int((df["parent_id"] > 0).sum()),
        "n_ambiguous": int(df["assignment_ambiguous"].sum()),
        "n_parents_with_objects": int(len(summary)),
    }


@tool(
    description="Distance from each object to the nearest reference surface (a Labels "
    "or mask layer), in µm — e.g. how far each synapse is from the membrane. Points "
    "sample the distance field at their location; label objects report their minimum "
    "boundary distance (closest approach), preserving per-object identity. With "
    "signed=True, distances inside the reference are negative.",
    phase="4",
    worker=True,
)
def measure_distance_to_reference(
    objects_layer: str,
    reference_layer: str,
    signed: bool = False,
    table_name: str | None = None,
) -> dict[str, Any]:
    from scipy import ndimage as ndi

    ref_snap = call_on_main(snapshot_layer, reference_layer)
    ref = materialize_array(ref_snap.data) > 0
    if ref.ndim not in (2, 3):
        raise ValueError("reference_layer must be a 2D/3D Labels/mask layer")
    if not ref.any():
        raise ValueError(f"reference layer {reference_layer!r} is empty")
    spacing = coords.layer_scale(ref_snap, ref.ndim)

    outside = ndi.distance_transform_edt(~ref, sampling=spacing)
    if signed:
        inside = ndi.distance_transform_edt(ref, sampling=spacing)
        field = outside - inside
    else:
        field = outside

    obj = _extract_objects(objects_layer)
    ids: list[int] = []
    dists: list[float] = []
    if obj.kind == "points":
        for oid, c in zip(obj.ids, obj.centroids):
            vox = tuple(int(round(v)) for v in c[: ref.ndim])
            if all(0 <= vox[a] < ref.shape[a] for a in range(ref.ndim)):
                ids.append(int(oid))
                dists.append(float(field[vox]))
    else:
        if obj.shape != tuple(ref.shape):
            raise ValueError("objects and reference must share a grid")
        for oid in obj.ids:
            vals = field[obj.labels == oid]
            if vals.size:
                ids.append(int(oid))
                # Closest approach: min unsigned distance (or most-inside if signed).
                dists.append(float(vals.min()))

    df = pd.DataFrame({"object_id": ids, "distance_um": dists})
    tname = table_name or f"{objects_layer}_dist_{reference_layer}"
    stored = _store(tname, df, {
        "op": "measure_distance_to_reference",
        "objects_layer": objects_layer,
        "reference_layer": reference_layer,
        "signed": bool(signed),
    })
    return {
        "table_name": stored,
        "n_objects": int(len(df)),
        "median_distance_um": float(df["distance_um"].median()) if len(df) else 0.0,
        "min_distance_um": float(df["distance_um"].min()) if len(df) else 0.0,
        "signed": bool(signed),
    }


@tool(
    description="Nearest-neighbour distances between object centroids, in µm — a "
    "clustering / dispersion readout. With other_layer given, measures each object's "
    "distance to the nearest object in the other set (e.g. red spots to green spots); "
    "otherwise within the same set (self excluded). Emits object_id + nn_distance_um.",
    phase="4",
    worker=True,
)
def nearest_neighbor_distances(
    objects_layer: str,
    other_layer: str | None = None,
    k: int = 1,
    table_name: str | None = None,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    obj = _extract_objects(objects_layer)
    src_world = coords.data_to_world(obj.centroids, obj.spacing) if len(obj.centroids) else np.empty((0, obj.ndim))

    if other_layer:
        other = _extract_objects(other_layer)
        tgt_world = coords.data_to_world(other.centroids, other.spacing) if len(other.centroids) else np.empty((0, other.ndim))
        self_query = False
    else:
        tgt_world = src_world
        self_query = True

    ids: list[int] = []
    nn: list[float] = []
    if len(src_world) and len(tgt_world):
        tree = cKDTree(tgt_world)
        kq = int(k) + (1 if self_query else 0)
        kq = min(kq, len(tgt_world))
        d, _idx = tree.query(src_world, k=kq)
        d = np.atleast_2d(d.T).T if d.ndim == 1 else d
        for i, oid in enumerate(obj.ids):
            row = np.atleast_1d(d[i])
            # Drop the zero self-match for within-set queries.
            row = row[1:] if self_query and len(row) > 1 else (row if not self_query else row[:0])
            if row.size:
                ids.append(int(oid))
                nn.append(float(row[: int(k)].mean()))

    df = pd.DataFrame({"object_id": ids, "nn_distance_um": nn})
    tname = table_name or f"{objects_layer}_nn"
    stored = _store(tname, df, {
        "op": "nearest_neighbor_distances",
        "objects_layer": objects_layer,
        "other_layer": other_layer,
        "k": int(k),
    })
    return {
        "table_name": stored,
        "n_objects": int(len(df)),
        "median_nn_um": float(df["nn_distance_um"].median()) if len(df) else 0.0,
        "mean_nn_um": float(df["nn_distance_um"].mean()) if len(df) else 0.0,
    }
