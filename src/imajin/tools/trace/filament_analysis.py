"""Filament analysis (Phase 5): dendrite/vessel diameter and rooted-tree topology.

``measure_filament_diameter`` samples the Euclidean distance transform of the
segmented mask along the skeleton to give a local radius/diameter profile (it
also fills the SWC radius column). ``compute_tree_topology`` derives centrifugal
branch order, Strahler number, and path length to the soma from the rooted tree
built by ``build_rooted_tree``.

Caveats (honest scope): the diameter is only as good as the segmentation and
inflates at junctions (junction-adjacent path ends are dropped from the summary).
Dendritic-spine detection is deliberately deferred — spine necks are often below
confocal resolution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imajin import session as state
from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import materialize_array
from imajin.tools._trace_store import _entry
from imajin.tools._trace_tables import _put_table
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


@tool(
    description="Measure filament diameter along a skeleton by sampling the distance "
    "transform of its segmented mask — a radius/diameter profile per node and per "
    "branch, for dendrite or vessel width. Uses the skeleton's source mask by default "
    "(or pass mask_layer). Junction-adjacent path ends are dropped from the branch "
    "summary because the distance transform inflates there. Also fills the SWC radius "
    "column. Diameter is only as reliable as the segmentation.",
    phase="6B",
    subagent="neural_tracer",
)
def measure_filament_diameter(
    skeleton_id: str,
    mask_layer: str | None = None,
) -> dict[str, Any]:
    import pandas as pd
    from scipy import ndimage as ndi

    entry = _entry(skeleton_id)
    skel = entry.skel
    spacing = np.asarray(entry.record.spacing, dtype=float)

    source = mask_layer or entry.record.mask_layer
    if not source:
        raise ValueError("no mask available; pass mask_layer (the segmented process mask)")
    snap = call_on_main(snapshot_layer, source)
    mask = materialize_array(snap.data) > 0
    if mask.shape != tuple(entry.skeleton_image.shape):
        raise ValueError(
            f"mask {mask.shape} must match the skeleton {tuple(entry.skeleton_image.shape)}"
        )

    edt = ndi.distance_transform_edt(mask, sampling=spacing)  # radius (µm) per voxel
    coords_idx = np.rint(np.asarray(skel.coordinates, dtype=float)).astype(int)
    radii = edt[tuple(coords_idx.T)]
    degrees = np.asarray(skel.degrees)

    node_df = pd.DataFrame({
        "node_id": np.arange(len(radii), dtype=int),
        "radius_um": radii.astype(float),
        "diameter_um": (2.0 * radii).astype(float),
        "degree": degrees.astype(int),
    })
    node_table = _put_table(
        f"{skeleton_id}_diameter_nodes", node_df,
        spec={"op": "measure_filament_diameter", "skeleton_id": skeleton_id, "level": "node"},
    )

    branch_rows = []
    interior_diams: list[float] = []
    for i in range(int(skel.n_paths)):
        pc = np.rint(np.asarray(skel.path_coordinates(i), dtype=float)).astype(int)
        vals = 2.0 * edt[tuple(pc.T)]
        interior = vals[1:-1] if len(vals) > 2 else vals  # drop junction/endpoint ends
        if len(interior):
            interior_diams.extend(interior.tolist())
            branch_rows.append({
                "branch_id": i,
                "mean_diameter_um": float(interior.mean()),
                "min_diameter_um": float(interior.min()),
                "max_diameter_um": float(interior.max()),
                "n_nodes": int(len(vals)),
            })
    branch_df = pd.DataFrame(
        branch_rows,
        columns=["branch_id", "mean_diameter_um", "min_diameter_um", "max_diameter_um", "n_nodes"],
    )
    branch_table = _put_table(
        f"{skeleton_id}_diameter_branches", branch_df,
        spec={"op": "measure_filament_diameter", "skeleton_id": skeleton_id, "level": "branch"},
    )

    # Persist per-node radii so the SWC export writes a real radius column.
    entry.record.parameters["node_radii_um"] = [float(r) for r in radii]
    entry.record.table_names["diameter_nodes"] = node_table
    entry.record.table_names["diameter_branches"] = branch_table

    median_d = float(np.median(interior_diams)) if interior_diams else 0.0
    state.put_qc_record(
        f"{skeleton_id}_diameter",
        status="pass",
        warnings=[],
        metrics={"kind": "filament_diameter", "median_diameter_um": median_d,
                 "n_branches": int(len(branch_df))},
    )
    return {
        "skeleton_id": skeleton_id,
        "node_table": node_table,
        "branch_table": branch_table,
        "median_diameter_um": median_d,
        "max_diameter_um": float(2.0 * radii.max()) if len(radii) else 0.0,
        "n_branches": int(len(branch_df)),
    }


@tool(
    description="Compute rooted-tree topology metrics from build_rooted_tree: "
    "centrifugal branch order, Strahler number, and path length to the soma per node. "
    "Requires build_rooted_tree to have been run first. Adds a topology table and "
    "reports the tree's maximum branch order, maximum Strahler order, leaf count, and "
    "total path length.",
    phase="6B",
    subagent="neural_tracer",
)
def compute_tree_topology(skeleton_id: str) -> dict[str, Any]:
    import pandas as pd

    entry = _entry(skeleton_id)
    tname = entry.record.table_names.get("rooted_tree")
    if not tname:
        raise ValueError("run build_rooted_tree first to produce the rooted-tree table")
    df = call_on_main(state.get_table, tname).copy()

    axis_cols = [c for c in ("z_um", "y_um", "x_um") if c in df.columns]
    coords = {int(r.node_id): np.array([getattr(r, c) for c in axis_cols], float)
              for r in df.itertuples()}
    parent = {int(r.node_id): int(r.parent_id) for r in df.itertuples()}
    children: dict[int, list[int]] = {n: [] for n in parent}
    root = None
    for node, par in parent.items():
        if par < 0:
            root = node
        else:
            children[par].append(node)

    order = df.sort_values("depth")["node_id"].astype(int).tolist()  # parents first

    path_len: dict[int, float] = {}
    branch_order: dict[int, int] = {}
    for n in order:
        par = parent[n]
        if par < 0:
            path_len[n] = 0.0
            branch_order[n] = 0
        else:
            path_len[n] = path_len[par] + float(np.linalg.norm(coords[n] - coords[par]))
            branch_order[n] = branch_order[par] + (1 if len(children[par]) >= 2 else 0)

    strahler: dict[int, int] = {}
    for n in reversed(order):  # children before parents
        kids = children[n]
        if not kids:
            strahler[n] = 1
        else:
            child_orders = [strahler[c] for c in kids]
            m = max(child_orders)
            strahler[n] = m + 1 if child_orders.count(m) >= 2 else m

    df["branch_order"] = df["node_id"].map(branch_order)
    df["strahler"] = df["node_id"].map(strahler)
    df["path_length_to_soma_um"] = df["node_id"].map(path_len)
    topo_table = _put_table(
        f"{skeleton_id}_topology", df,
        spec={"op": "compute_tree_topology", "skeleton_id": skeleton_id},
    )
    entry.record.table_names["topology"] = topo_table

    n_leaves = int(sum(1 for n in children if not children[n]))
    result = {
        "skeleton_id": skeleton_id,
        "table_name": topo_table,
        "root_node": int(root) if root is not None else -1,
        "max_branch_order": int(max(branch_order.values())) if branch_order else 0,
        "max_strahler": int(max(strahler.values())) if strahler else 0,
        "n_leaves": n_leaves,
        "total_path_length_um": float(max(path_len.values())) if path_len else 0.0,
    }
    state.put_qc_record(
        f"{skeleton_id}_topology", status="pass", warnings=[],
        metrics={"kind": "tree_topology", **result},
    )
    return result
