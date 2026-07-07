"""Connectivity-aware filament tracing (Phase 4, split into two stages).

``skeletonize`` already extracts a skan graph (stage 4a). This module adds the
two stages that turn a fragmented skeleton into a valid neuron tree:

- ``propose_filament_bridges`` (4b) — proposes joins between disconnected
  skeleton components, **evidence-gated** (gap length, endpoint tangent
  continuity, intensity/vesselness support along the candidate segment), and
  writes a reviewable bridge QC table. It does not mutate the skeleton.
- ``build_rooted_tree`` (4c) — builds a directed rooted tree from the skan graph
  plus accepted bridges, rooted at the soma (or an endpoint), with deterministic
  parent ordering and an explicit dropped-component policy.

Honest scope: this is a skeleton-graph tracer, not an ML tracer. It cannot undo
false merges created at true neurite crossings (the binary skeleton shares those
pixels) — see the plan's documented failure modes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from imajin import session as state
from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import materialize_array
from imajin.tools._trace_store import _entry
from imajin.tools._trace_tables import _put_table
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


def _endpoint_neighbor(graph, node: int) -> int:
    start, end = graph.indptr[node], graph.indptr[node + 1]
    nbrs = graph.indices[start:end]
    return int(nbrs[0]) if len(nbrs) else int(node)


@tool(
    description="Propose bridges across gaps in a skeleton to reconnect a filament "
    "broken by faint or missing signal. Candidate joins between endpoints of "
    "different components are gated by gap length (<= max_gap_um), endpoint tangent "
    "continuity (the bridge must continue the process, not double back), and "
    "intensity/vesselness support sampled along the segment (from support_layer, "
    "default the skeleton's source image). Writes a reviewable bridge table "
    "(accepted + reason) and stores accepted bridges on the trace for build_rooted_tree. "
    "Does not modify the skeleton.",
    phase="6B",
    subagent="neural_tracer",
)
def propose_filament_bridges(
    skeleton_id: str,
    max_gap_um: float,
    support_layer: str | None = None,
    min_support: float = 0.2,
    max_tangent_angle_deg: float = 60.0,
) -> dict[str, Any]:
    import pandas as pd
    from scipy.sparse.csgraph import connected_components

    entry = _entry(skeleton_id)
    skel = entry.skel
    spacing = np.asarray(entry.record.spacing, dtype=float)
    graph = skel.graph.tocsr()
    coords_idx = np.asarray(skel.coordinates, dtype=float)
    coords_um = coords_idx * spacing
    degrees = np.asarray(skel.degrees)
    _ncomp, comp = connected_components(graph, directed=False)
    endpoints = np.where(degrees == 1)[0]

    tangents: dict[int, np.ndarray] = {}
    for e in endpoints:
        nb = _endpoint_neighbor(graph, int(e))
        v = coords_um[e] - coords_um[nb]
        norm = np.linalg.norm(v)
        tangents[int(e)] = v / norm if norm > 0 else np.zeros(coords_um.shape[1])

    support = None
    if support_layer is not None:
        snap = call_on_main(snapshot_layer, support_layer)
        arr = materialize_array(snap.data).astype(np.float32)
        if arr.shape == tuple(entry.skeleton_image.shape):
            peak = float(arr.max())
            support = arr / peak if peak > 0 else arr

    cos_thresh = math.cos(math.radians(float(max_tangent_angle_deg)))
    candidates: list[dict[str, Any]] = []
    from skimage.draw import line_nd

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a, b = int(endpoints[i]), int(endpoints[j])
            if comp[a] == comp[b]:
                continue
            gap = float(np.linalg.norm(coords_um[a] - coords_um[b]))
            if gap > float(max_gap_um):
                continue
            bd = coords_um[b] - coords_um[a]
            bnorm = np.linalg.norm(bd)
            bdir = bd / bnorm if bnorm > 0 else np.zeros_like(bd)
            ca = float(np.dot(tangents[a], bdir))
            cb = float(np.dot(tangents[b], -bdir))
            tangent_score = min(ca, cb)
            support_score = 1.0
            if support is not None:
                vox = line_nd(coords_idx[a], coords_idx[b], endpoint=True)
                support_score = float(support[vox].min())
            tangent_ok = tangent_score >= cos_thresh
            support_ok = support_score >= float(min_support)
            reason = "accepted" if (tangent_ok and support_ok) else (
                "tangent" if not tangent_ok else "support"
            )
            candidates.append({
                "node_a": a, "node_b": b,
                "component_a": int(comp[a]), "component_b": int(comp[b]),
                "gap_um": gap, "tangent_score": tangent_score,
                "support_score": support_score,
                "accepted": bool(tangent_ok and support_ok), "reason": reason,
                "_rank": 0.5 * tangent_score + 0.5 * support_score,
            })

    # Greedy one-bridge-per-endpoint matching among accepted candidates.
    accepted: list[tuple[int, int]] = []
    used: set[int] = set()
    for c in sorted(
        [c for c in candidates if c["accepted"]], key=lambda c: c["_rank"], reverse=True
    ):
        if c["node_a"] in used or c["node_b"] in used:
            c["accepted"] = False
            c["reason"] = "endpoint_taken"
            continue
        used.add(c["node_a"])
        used.add(c["node_b"])
        accepted.append((c["node_a"], c["node_b"]))

    rows = []
    for k, c in enumerate(candidates):
        c.pop("_rank", None)
        rows.append({"bridge_id": k, **c})
    df = pd.DataFrame(
        rows,
        columns=[
            "bridge_id", "node_a", "node_b", "component_a", "component_b",
            "gap_um", "tangent_score", "support_score", "accepted", "reason",
        ],
    )
    table_name = _put_table(
        f"{skeleton_id}_bridges", df,
        spec={"op": "propose_filament_bridges", "skeleton_id": skeleton_id,
              "max_gap_um": float(max_gap_um)},
    )
    entry.record.parameters["bridges"] = [list(p) for p in accepted]
    entry.record.table_names["bridges"] = table_name

    warnings: list[str] = []
    if support is None and support_layer is not None:
        warnings.append("support_layer shape did not match the skeleton; support gate skipped")
    state.put_qc_record(
        f"{skeleton_id}_bridges",
        status="pass" if accepted else "warning",
        warnings=warnings,
        metrics={"kind": "filament_bridges", "n_candidates": len(candidates),
                 "n_accepted": len(accepted), "n_endpoints": int(len(endpoints))},
    )
    return {
        "skeleton_id": skeleton_id,
        "table_name": table_name,
        "n_candidates": len(candidates),
        "n_accepted": len(accepted),
        "n_endpoints": int(len(endpoints)),
        "warnings": warnings,
    }


@tool(
    description="Build a directed rooted tree from a skeleton plus any accepted bridges "
    "(from propose_filament_bridges), rooted at the soma if set (else an endpoint). "
    "Breaks cycles by a deterministic breadth-first spanning tree and assigns each node "
    "a parent, so SWC parent pointers are valid. Components not reachable from the root "
    "(no bridge) are dropped and reported. Stores a rooted-tree table (node_id, "
    "parent_id, depth, via_bridge, µm coordinates).",
    phase="6B",
    subagent="neural_tracer",
)
def build_rooted_tree(
    skeleton_id: str,
    apply_bridges: bool = True,
) -> dict[str, Any]:
    import pandas as pd

    from collections import deque

    import networkx as nx

    entry = _entry(skeleton_id)
    skel = entry.skel
    spacing = np.asarray(entry.record.spacing, dtype=float)
    coords_idx = np.asarray(skel.coordinates, dtype=float)
    coords_um = coords_idx * spacing
    degrees = np.asarray(skel.degrees)
    ndim = coords_um.shape[1]

    G = nx.Graph()
    G.add_nodes_from(range(len(coords_um)))
    coo = skel.graph.tocoo()
    for s, d, w in zip(coo.row, coo.col, coo.data):
        if int(s) < int(d):
            G.add_edge(int(s), int(d), weight=float(w), via_bridge=False)
    bridges = entry.record.parameters.get("bridges", []) if apply_bridges else []
    for a, b in bridges:
        gap = float(np.linalg.norm(coords_um[int(a)] - coords_um[int(b)]))
        G.add_edge(int(a), int(b), weight=gap, via_bridge=True)

    if entry.record.soma is not None:
        soma = np.asarray(entry.record.soma, dtype=float)
        root = int(np.argmin(np.linalg.norm(coords_um - soma, axis=1)))
    else:
        endpoints = np.where(degrees == 1)[0]
        root = int(endpoints[0]) if len(endpoints) else 0

    # Deterministic BFS spanning tree (sorted neighbours) → parent pointers.
    parent = {root: -1}
    depth = {root: 0}
    via = {root: False}
    q: deque[int] = deque([root])
    while q:
        u = q.popleft()
        for v in sorted(G.neighbors(u)):
            if v not in parent:
                parent[v] = u
                depth[v] = depth[u] + 1
                via[v] = bool(G[u][v]["via_bridge"])
                q.append(v)

    tree_nodes = sorted(parent.keys())
    axis_names = ("z", "y", "x") if ndim == 3 else ("y", "x")
    data: dict[str, Any] = {
        "node_id": tree_nodes,
        "parent_id": [parent[n] for n in tree_nodes],
        "depth": [depth[n] for n in tree_nodes],
        "via_bridge": [via[n] for n in tree_nodes],
    }
    for i, ax in enumerate(axis_names):
        data[f"{ax}_um"] = [float(coords_um[n, i]) for n in tree_nodes]
    df = pd.DataFrame(data)
    table_name = _put_table(
        f"{skeleton_id}_rooted_tree", df,
        spec={"op": "build_rooted_tree", "skeleton_id": skeleton_id,
              "apply_bridges": bool(apply_bridges)},
    )
    entry.record.table_names["rooted_tree"] = table_name

    from scipy.sparse.csgraph import connected_components

    _ncomp, comp = connected_components(skel.graph, directed=False)
    merged = len({int(comp[n]) for n in tree_nodes})
    n_dropped = len(coords_um) - len(tree_nodes)
    n_bridges = int(sum(1 for n in tree_nodes if via[n]))
    warnings: list[str] = []
    if n_dropped:
        warnings.append(f"{n_dropped} node(s) in unbridged components dropped from the tree")
    state.put_qc_record(
        f"{skeleton_id}_rooted_tree",
        status="warning" if warnings else "pass",
        warnings=warnings,
        metrics={"kind": "rooted_tree", "n_tree_nodes": len(tree_nodes),
                 "n_dropped": int(n_dropped), "n_bridges_applied": n_bridges,
                 "n_components_merged": int(merged)},
    )
    return {
        "skeleton_id": skeleton_id,
        "table_name": table_name,
        "root_node": int(root),
        "n_tree_nodes": len(tree_nodes),
        "n_total_nodes": int(len(coords_um)),
        "n_dropped": int(n_dropped),
        "n_bridges_applied": n_bridges,
        "n_components_merged": int(merged),
        "warnings": warnings,
    }
