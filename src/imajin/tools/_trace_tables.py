from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.qt_dispatch import call_on_main
from imajin.session import put_table


_BRANCH_TYPES = {
    0: "endpoint-endpoint",
    1: "junction-endpoint",
    2: "junction-junction",
    3: "isolated-cycle",
}


def _put_table(name: str, df: pd.DataFrame, spec: dict[str, Any]) -> str:
    return call_on_main(put_table, name, df, spec=spec)


def _scale_tuple(scale: tuple[float, ...] | None, ndim: int) -> tuple[float, ...]:
    if not scale:
        return (1.0,) * ndim
    values = tuple(float(v) for v in scale[:ndim])
    if len(values) < ndim:
        values = values + (1.0,) * (ndim - len(values))
    return values


def _scale_is_physical(spacing: tuple[float, ...]) -> bool:
    return any(abs(v - 1.0) > 1e-9 for v in spacing)


def _normalize_branch_df(df: pd.DataFrame, spacing: tuple[float, ...]) -> pd.DataFrame:
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
    if "branch_length" in df.columns:
        df["branch_length_scaled"] = df["branch_length"].astype(float)
        if _scale_is_physical(spacing):
            df["branch_length_um"] = df["branch_length"].astype(float)
    if "euclidean_distance" in df.columns:
        df["euclidean_distance_scaled"] = df["euclidean_distance"].astype(float)
        if _scale_is_physical(spacing):
            df["euclidean_distance_um"] = df["euclidean_distance"].astype(float)
    if "branch_length" in df.columns and "euclidean_distance" in df.columns:
        ed = df["euclidean_distance"].replace(0, np.nan)
        df["tortuosity"] = df["branch_length"] / ed
    df.insert(0, "branch_id", np.arange(len(df), dtype=int))
    return df


def _branch_summary(skel: Any, spacing: tuple[float, ...]) -> pd.DataFrame:
    from skan import summarize

    return _normalize_branch_df(summarize(skel, separator="-"), spacing)


def _node_table(skel: Any, spacing: tuple[float, ...]) -> pd.DataFrame:
    from scipy.sparse.csgraph import connected_components

    coords = np.asarray(skel.coordinates)
    physical = coords.astype(float) * np.asarray(spacing, dtype=float)
    n_components, labels = connected_components(skel.graph, directed=False)
    data: dict[str, Any] = {
        "node_id": np.arange(len(coords), dtype=int),
        "degree": np.asarray(skel.degrees, dtype=int),
        "component_id": labels.astype(int),
    }
    for axis in range(coords.shape[1]):
        data[f"image_coord_{axis}"] = coords[:, axis].astype(int)
        data[f"coord_{axis}_scaled"] = physical[:, axis].astype(float)
        if _scale_is_physical(spacing):
            data[f"coord_{axis}_um"] = physical[:, axis].astype(float)
    df = pd.DataFrame(data)
    df.attrs["n_components"] = int(n_components)
    return df


def _edge_table(skel: Any, spacing: tuple[float, ...]) -> pd.DataFrame:
    graph = skel.graph.tocoo()
    rows: list[dict[str, Any]] = []
    for src, dst, dist in zip(graph.row, graph.col, graph.data, strict=False):
        if int(src) >= int(dst):
            continue
        row = {
            "edge_id": len(rows),
            "node_id_src": int(src),
            "node_id_dst": int(dst),
            "edge_length_scaled": float(dist),
        }
        if _scale_is_physical(spacing):
            row["edge_length_um"] = float(dist)
        rows.append(row)
    return pd.DataFrame(rows)


def _component_table(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if nodes.empty:
        return pd.DataFrame(columns=["component_id", "n_nodes", "n_edges"])
    for component_id, group in nodes.groupby("component_id"):
        node_ids = set(int(v) for v in group["node_id"].tolist())
        if edges.empty:
            component_edges = edges
        else:
            component_edges = edges[
                edges["node_id_src"].isin(node_ids) & edges["node_id_dst"].isin(node_ids)
            ]
        row: dict[str, Any] = {
            "component_id": int(component_id),
            "n_nodes": int(len(group)),
            "n_edges": int(len(component_edges)),
        }
        for col in [c for c in group.columns if c.startswith("coord_") and c.endswith("_scaled")]:
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def store_graph_tables(
    skeleton_id: str,
    skel: Any,
    spacing: tuple[float, ...],
) -> tuple[dict[str, str], int]:
    nodes = _node_table(skel, spacing)
    edges = _edge_table(skel, spacing)
    components = _component_table(nodes, edges)
    names = {
        "nodes": _put_table(
            f"{skeleton_id}_nodes",
            nodes,
            spec={"op": "skeleton_nodes", "skeleton_id": skeleton_id},
        ),
        "edges": _put_table(
            f"{skeleton_id}_edges",
            edges,
            spec={"op": "skeleton_edges", "skeleton_id": skeleton_id},
        ),
        "components": _put_table(
            f"{skeleton_id}_components",
            components,
            spec={"op": "skeleton_components", "skeleton_id": skeleton_id},
        ),
    }
    return names, int(len(components))
