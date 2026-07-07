from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imajin.session import get_layer, get_viewer


@dataclass(frozen=True)
class LayerSnapshot:
    name: str
    data: Any
    scale: tuple[float, ...]
    metadata: dict[str, Any]
    kind: str = ""


def _layer_kind(layer: Any) -> str:
    k = getattr(layer, "kind", None) or getattr(layer, "_type_string", None)
    if isinstance(k, str):
        return k.lower()
    return type(layer).__name__.lower()


def snapshot_layer(name: str) -> LayerSnapshot:

    layer = get_layer(name)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    source = getattr(layer, "source", None)
    source_path = getattr(source, "path", None) if source is not None else None
    if source_path and not (metadata.get("source_path") or metadata.get("path")):
        metadata["source_path"] = str(source_path)
        metadata["path"] = str(source_path)
    return LayerSnapshot(
        name=layer.name,
        data=layer.data,
        scale=tuple(float(s) for s in getattr(layer, "scale", ())),
        metadata=metadata,
        kind=_layer_kind(layer),
    )


def add_image_from_worker(
    data: Any,
    *,
    name: str,
    scale: tuple[float, ...],
    metadata: dict[str, Any],
    **kwargs: Any,
):

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

    return get_viewer().add_labels(
        data,
        name=name,
        scale=scale or None,
        metadata=metadata,
    )


def add_points_from_worker(
    data: Any,
    *,
    name: str,
    scale: tuple[float, ...],
    metadata: dict[str, Any],
    **kwargs: Any,
):
    """Add a Points layer holding detections in data/index coordinates (the
    canonical layer geometry frame — see ``analysis/coords.py``). napari applies
    ``scale`` for rendering; the points array itself stays unscaled."""

    return get_viewer().add_points(
        data,
        name=name,
        scale=scale or None,
        metadata=metadata,
        **kwargs,
    )
