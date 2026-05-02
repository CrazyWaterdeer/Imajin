from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imajin.agent import state
from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_layer, get_viewer, put_table
from imajin.tools.napari_ops import (
    add_image_from_worker,
    add_labels_from_worker,
    snapshot_layer,
)
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


_BRANCH_TYPES = {
    0: "endpoint-endpoint",
    1: "junction-endpoint",
    2: "junction-junction",
    3: "isolated-cycle",
}

_TRACE_STATUSES = {"raw", "reviewed", "pruned", "exported"}
_BRANCH_QC_STATUSES = {"accepted", "rejected", "not_checked"}


@dataclass
class NeuralTraceRecord:
    trace_id: str
    source_layer: str
    mask_layer: str | None
    skeleton_layer: str
    spacing: tuple[float, ...]
    units: tuple[str, ...] | None = None
    status: str = "raw"
    parameters: dict[str, Any] = field(default_factory=dict)
    n_paths: int = 0
    n_components: int = 0
    table_names: dict[str, str] = field(default_factory=dict)
    parent_trace_id: str | None = None
    soma: tuple[float, ...] | None = None
    region: str | int | None = None


@dataclass
class NeuralTraceQC:
    trace_id: str
    accepted: bool | None = None
    rejected_branch_ids: list[int] = field(default_factory=list)
    notes: str | None = None
    branch_statuses: dict[int, str] = field(default_factory=dict)
    branch_reasons: dict[int, str] = field(default_factory=dict)


@dataclass
class _SkeletonEntry:
    skel: Any
    skeleton_image: np.ndarray
    record: NeuralTraceRecord
    qc: NeuralTraceQC


_SKELETON_REGISTRY: dict[str, _SkeletonEntry] = {}


def get_skeleton(skel_id: str):
    return _entry(skel_id).skel


def get_trace_record(skel_id: str) -> NeuralTraceRecord:
    return _entry(skel_id).record


def list_trace_records() -> list[dict[str, Any]]:
    return [
        {**asdict(entry.record), "qc": asdict(entry.qc)}
        for entry in _SKELETON_REGISTRY.values()
    ]


def reset_skeletons() -> None:
    _SKELETON_REGISTRY.clear()


def _entry(skel_id: str) -> _SkeletonEntry:
    if skel_id not in _SKELETON_REGISTRY:
        raise KeyError(f"skeleton {skel_id!r} not found. Available: {list(_SKELETON_REGISTRY)}")
    return _SKELETON_REGISTRY[skel_id]


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


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from skimage.measure import label

    labeled = label(mask.astype(bool), connectivity=1)
    return labeled.astype(np.int32), int(labeled.max())


def _layer_kind(layer_name: str) -> str:
    try:
        layer = call_on_main(get_layer, layer_name)
    except Exception:
        return ""
    kind = getattr(layer, "kind", None)
    if isinstance(kind, str):
        return kind.lower()
    return type(layer).__name__.lower()


def _binary_from_layer_data(
    data: np.ndarray,
    *,
    layer_name: str,
    threshold: float | None,
) -> np.ndarray:
    kind = _layer_kind(layer_name)
    if "label" in kind:
        return data > 0
    if threshold is not None:
        return data > float(threshold)
    finite = data[np.isfinite(data)]
    unique = np.unique(finite)
    if unique.size <= 2 and set(unique.tolist()).issubset({0, 1, False, True}):
        return data.astype(bool)
    raise ValueError(
        "skeletonize expects a binary/Labels layer. For continuous image data, "
        "run segment_neural_processes first or pass an explicit threshold."
    )


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


def _store_graph_tables(skeleton_id: str) -> dict[str, str]:
    entry = _entry(skeleton_id)
    nodes = _node_table(entry.skel, entry.record.spacing)
    edges = _edge_table(entry.skel, entry.record.spacing)
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
    entry.record.table_names.update(names)
    entry.record.n_components = int(len(components))
    return names


def _register_skeleton(
    *,
    skel: Any,
    skeleton_image: np.ndarray,
    source_layer: str,
    mask_layer: str | None,
    skeleton_layer: str,
    spacing: tuple[float, ...],
    parameters: dict[str, Any],
    status: str = "raw",
    parent_trace_id: str | None = None,
) -> str:
    skel_id = f"skel_{len(_SKELETON_REGISTRY)}_{source_layer}"
    record = NeuralTraceRecord(
        trace_id=skel_id,
        source_layer=source_layer,
        mask_layer=mask_layer,
        skeleton_layer=skeleton_layer,
        spacing=spacing,
        units=tuple("um" for _ in spacing) if _scale_is_physical(spacing) else None,
        status=status,
        parameters=dict(parameters),
        n_paths=int(skel.n_paths),
        parent_trace_id=parent_trace_id,
    )
    _SKELETON_REGISTRY[skel_id] = _SkeletonEntry(
        skel=skel,
        skeleton_image=skeleton_image.astype(bool),
        record=record,
        qc=NeuralTraceQC(trace_id=skel_id),
    )
    _store_graph_tables(skel_id)
    return skel_id


def _normalize_image(data: np.ndarray) -> np.ndarray:
    from skimage.exposure import rescale_intensity

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return rescale_intensity(data, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def _rolling_ball_subtract(data: np.ndarray, radius: float = 50.0) -> np.ndarray:
    from skimage.restoration import rolling_ball

    if data.ndim == 2:
        return data - rolling_ball(data, radius=radius)
    if data.ndim == 3:
        out = np.empty_like(data)
        for z in range(data.shape[0]):
            out[z] = data[z] - rolling_ball(data[z], radius=radius)
        return out
    raise ValueError(f"Expected 2D or 3D layer, got shape {data.shape}")


@tool(
    description="Enhance a 2D/3D neural process image before segmentation. Methods: "
    "tubeness/sato, frangi, gaussian, or none. Optionally subtract rolling-ball "
    "background and percentile-normalize. Adds a new image layer without mutating raw data.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def enhance_neural_processes(
    layer: str,
    method: str = "tubeness",
    sigma: float | tuple[float, ...] | None = None,
    background: str | None = "rolling_ball",
    normalize: bool = True,
) -> dict[str, Any]:
    from skimage.filters import frangi, gaussian, sato

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data).astype(np.float32, copy=False)
    if data.ndim not in (2, 3):
        raise ValueError(f"enhance_neural_processes expects 2D or 3D data, got {data.shape}")

    out = data
    if background in {"rolling_ball", "rolling-ball"}:
        out = _rolling_ball_subtract(out, radius=50.0)
    elif background not in {None, "none", ""}:
        raise ValueError("background must be rolling_ball, none, or None")

    method_key = method.lower().strip()
    if sigma is None:
        sigmas = (1.0, 2.0, 3.0)
        gaussian_sigma: float | tuple[float, ...] = 1.0
    elif isinstance(sigma, (list, tuple)):
        sigmas = tuple(float(s) for s in sigma)
        gaussian_sigma = tuple(float(s) for s in sigma)
    else:
        sigmas = (float(sigma),)
        gaussian_sigma = float(sigma)

    if method_key in {"none", "raw"}:
        enhanced = out
    elif method_key in {"gaussian", "denoise"}:
        enhanced = gaussian(out, sigma=gaussian_sigma, preserve_range=True)
    elif method_key in {"tubeness", "sato"}:
        enhanced = sato(out, sigmas=sigmas, black_ridges=False)
    elif method_key in {"frangi", "vesselness"}:
        enhanced = frangi(out, sigmas=sigmas, black_ridges=False)
    else:
        raise ValueError("method must be tubeness, sato, frangi, vesselness, gaussian, or none")

    enhanced = enhanced.astype(np.float32, copy=False)
    if normalize:
        enhanced = _normalize_image(enhanced)

    new = call_on_main(
        add_image_from_worker,
        enhanced,
        name=f"{L.name}_neural_enhanced",
        scale=tuple(L.scale),
        metadata={
            **dict(L.metadata or {}),
            "source_layer": L.name,
            "op": "enhance_neural_processes",
            "method": method_key,
            "sigma": sigma,
            "background": background,
            "normalize": normalize,
        },
        colormap="gray",
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in enhanced.shape),
        "dtype": str(enhanced.dtype),
        "scale": tuple(float(s) for s in L.scale),
        "method": method_key,
    }


@tool(
    description="Threshold a 2D/3D enhanced neural process image into connected process "
    "labels. Threshold may be otsu, yen, triangle, local/adaptive, or a numeric scalar.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def segment_neural_processes(
    layer: str,
    threshold: str | float = "otsu",
    min_size_um3: float | None = None,
    fill_holes: bool = False,
    keep_largest: bool = False,
) -> dict[str, Any]:
    from scipy import ndimage as ndi
    from skimage.filters import threshold_local, threshold_otsu, threshold_triangle, threshold_yen
    from skimage.morphology import remove_small_objects

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    if data.ndim not in (2, 3):
        raise ValueError(f"segment_neural_processes expects 2D or 3D data, got {data.shape}")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("cannot segment an image with no finite pixels")

    mode = str(threshold).lower().strip()
    if isinstance(threshold, (int, float)):
        thr = float(threshold)
        mask = data > thr
        threshold_value: float | str = thr
    elif mode == "otsu":
        thr = float(threshold_otsu(finite))
        mask = data > thr
        threshold_value = thr
    elif mode == "yen":
        thr = float(threshold_yen(finite))
        mask = data > thr
        threshold_value = thr
    elif mode == "triangle":
        thr = float(threshold_triangle(finite))
        mask = data > thr
        threshold_value = thr
    elif mode in {"local", "adaptive"}:
        block = max(3, min(31, *(s for s in data.shape[-2:])))
        if block % 2 == 0:
            block -= 1
        if data.ndim == 2:
            local = threshold_local(data, block_size=block)
            mask = data > local
        else:
            mask = np.stack(
                [plane > threshold_local(plane, block_size=block) for plane in data],
                axis=0,
            )
        threshold_value = f"local_block_{block}"
    else:
        try:
            thr = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "threshold must be otsu, yen, triangle, local/adaptive, or numeric"
            ) from exc
        mask = data > thr
        threshold_value = thr

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    spacing = _scale_tuple(tuple(L.scale), data.ndim)
    min_size_pixels = 0
    if min_size_um3 is not None and min_size_um3 > 0:
        voxel_measure = float(np.prod(spacing)) if _scale_is_physical(spacing) else 1.0
        min_size_pixels = max(1, int(np.ceil(float(min_size_um3) / voxel_measure)))
        try:
            mask = remove_small_objects(
                mask.astype(bool), max_size=max(0, min_size_pixels - 1)
            )
        except TypeError:
            mask = remove_small_objects(mask.astype(bool), min_size=min_size_pixels)

    labels, component_count = _component_labels(mask)
    if keep_largest and component_count > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        labels = (labels == largest).astype(np.int32)
        component_count = 1
    else:
        labels = labels.astype(np.int32)

    foreground_fraction = float(np.count_nonzero(labels) / labels.size)
    warnings: list[str] = []
    if foreground_fraction < 0.0005:
        warnings.append("Foreground fraction is very low; threshold may be too stringent.")
    if foreground_fraction > 0.35:
        warnings.append("Foreground fraction is high for sparse neural process tracing.")
    if component_count == 0:
        warnings.append("No connected process components were found.")

    new = call_on_main(
        add_labels_from_worker,
        labels,
        name=f"{L.name}_process_mask",
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "op": "segment_neural_processes",
            "threshold": threshold,
            "threshold_value": threshold_value,
            "min_size_pixels": min_size_pixels,
            "foreground_fraction": foreground_fraction,
            "component_count": component_count,
            "warnings": warnings,
        },
    )
    status = "fail" if component_count == 0 else ("warning" if warnings else "pass")
    state.put_qc_record(
        new.name,
        status=status,  # type: ignore[arg-type]
        warnings=warnings,
        metrics={
            "kind": "neural_process_segmentation",
            "foreground_fraction": foreground_fraction,
            "component_count": component_count,
            "threshold_value": threshold_value,
        },
    )
    return {
        "mask_layer": new.name,
        "shape": tuple(int(s) for s in labels.shape),
        "foreground_fraction": foreground_fraction,
        "component_count": component_count,
        "threshold_value": threshold_value,
        "min_size_pixels": min_size_pixels,
        "warnings": warnings,
        "qc_status": status,
    }


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


def _swc_coordinates(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] == 2:
        y = coords[:, 0]
        x = coords[:, 1]
        z = np.zeros(len(coords), dtype=float)
        return np.column_stack([x, y, z])
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    return np.column_stack([x, y, z])


def _write_swc(entry: _SkeletonEntry, path: Path) -> None:
    graph = entry.skel.graph.tocsr()
    n = graph.shape[0]
    coords = np.asarray(entry.skel.coordinates, dtype=float) * np.asarray(entry.record.spacing)
    swc_coords = _swc_coordinates(coords)

    degrees = np.asarray(entry.skel.degrees)
    endpoints = np.where(degrees == 1)[0]
    if entry.record.soma is not None:
        soma = np.asarray(entry.record.soma, dtype=float)
        root = int(np.argmin(np.linalg.norm(coords - soma, axis=1)))
    elif len(endpoints):
        root = int(endpoints[0])
    else:
        root = 0

    parent = np.full(n, -2, dtype=int)
    parent[root] = -1
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        start, end = graph.indptr[node], graph.indptr[node + 1]
        for nb in graph.indices[start:end]:
            nb = int(nb)
            if parent[nb] != -2:
                continue
            parent[nb] = node
            queue.append(nb)
    disconnected = np.where(parent == -2)[0]
    for node in disconnected:
        parent[node] = -1

    lines = [
        "# imajin SWC export",
        "# type 3 is used for all process nodes; radius is a placeholder.",
        "# If no soma was annotated, the root is the first endpoint or node.",
    ]
    for node_id in range(n):
        x, y, z = swc_coords[node_id]
        lines.append(
            f"{node_id + 1} 3 {x:.6f} {y:.6f} {z:.6f} 0.5 "
            f"{-1 if parent[node_id] < 0 else parent[node_id] + 1}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    out = Path(output_path).expanduser().resolve()
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
    description="Query a connectome database for nearest reference neurons by morphology. "
    "STUB — returns 'not implemented' until a target organism / database backend is "
    "wired up (planned: FlyWire, neuPrint, navis/NBLAST plugins).",
    phase="6B",
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
        "note": "Connectome backend deferred. Local trace/export is available; reference matching needs a plugin/backend.",
    }


@tool(
    description="Classify a skeleton's neuron type by comparison to a reference set. "
    "STUB — returns 'not implemented' until a reference morphology DB or learned "
    "classifier is added.",
    phase="6B",
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
        "note": "Classification deferred. Use local morphology descriptors/export until a reference backend exists.",
    }
