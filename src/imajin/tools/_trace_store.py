from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from imajin.tools._trace_tables import _scale_is_physical, store_graph_tables


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


def _store_graph_tables(skeleton_id: str) -> dict[str, str]:
    entry = _entry(skeleton_id)
    names, n_components = store_graph_tables(
        skeleton_id,
        entry.skel,
        entry.record.spacing,
    )
    entry.record.table_names.update(names)
    entry.record.n_components = n_components
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
