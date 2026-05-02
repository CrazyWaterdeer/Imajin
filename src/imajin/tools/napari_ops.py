from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerSnapshot:
    name: str
    data: Any
    scale: tuple[float, ...]
    metadata: dict[str, Any]


def snapshot_layer(name: str) -> LayerSnapshot:
    from imajin.agent.state import get_layer

    layer = get_layer(name)
    return LayerSnapshot(
        name=layer.name,
        data=layer.data,
        scale=tuple(float(s) for s in getattr(layer, "scale", ())),
        metadata=dict(getattr(layer, "metadata", {}) or {}),
    )


def add_image_from_worker(
    data: Any,
    *,
    name: str,
    scale: tuple[float, ...],
    metadata: dict[str, Any],
    **kwargs: Any,
):
    from imajin.agent.state import get_viewer

    viewer = get_viewer()
    return viewer.add_image(
        data,
        name=name,
        scale=scale or None,
        metadata=metadata,
        **kwargs,
    )


def add_labels_from_worker(
    data: Any,
    *,
    name: str,
    scale: tuple[float, ...],
    metadata: dict[str, Any],
):
    from imajin.agent.state import get_viewer

    return get_viewer().add_labels(
        data,
        name=name,
        scale=scale or None,
        metadata=metadata,
    )
