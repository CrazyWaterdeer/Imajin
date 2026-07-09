"""Shared layer-loading helpers for tools.

Centralises the ``snapshot the layer on the main thread + materialize its data``
step that the tool functions repeat.
"""

from __future__ import annotations

from typing import Any

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import materialize_array
from imajin.tools.napari_ops import snapshot_layer


def snapshot_array(name: str, *, dtype: Any = None) -> tuple[Any, Any]:
    """Snapshot layer ``name`` on the main thread and materialize its data.

    Returns ``(snapshot, ndarray)`` — the snapshot carries ``scale`` / ``metadata``
    / ``kind`` for callers that need physical units or the layer type.
    """
    snap = call_on_main(snapshot_layer, name)
    return snap, materialize_array(snap.data, dtype=dtype)
