from __future__ import annotations

from typing import Any

from imajin.agent.state import (
    AmbiguousChannelError,
    canonical_channel_color,
    canonical_channel_role,
    list_channel_annotations,
    put_channel_annotation,
    resolve_layer_name,
    resolve_target_channel,
)
from imajin.tools.registry import tool


@tool(
    description="Annotate an image layer's channel identity. Keep this simple: role is "
    "target, counterstain, or ignore. Color understands green, red, UV, and IR/far red "
    "aliases. The target channel is the default channel for segmentation, intensity, "
    "cell size, and time-course measurement.",
    phase="1.5",
)
def annotate_channel(
    layer: str,
    role: str = "target",
    color: str | None = None,
    marker: str | None = None,
    biological_target: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    layer_name = put_channel_annotation(
        layer_name=layer,
        role=role,
        color=color,
        marker=marker,
        biological_target=biological_target,
        notes=notes,
    )
    return {
        "layer": layer_name,
        "role": canonical_channel_role(role),
        "color": canonical_channel_color(color),
        "marker": marker,
        "biological_target": biological_target,
        "notes": notes,
    }


@tool(
    name="list_channel_annotations",
    description="List current channel annotations: target/counterstain/ignore role, "
    "canonical color (green/red/uv/ir), marker, and biological target.",
    phase="1.5",
)
def list_channel_annotations_tool() -> list[dict[str, Any]]:
    return list_channel_annotations()


@tool(
    name="resolve_channel",
    description="Resolve a human channel description such as green, red, UV, IR, far "
    "red, GFP, GCaMP, DAPI, RFP, or a marker name to the matching napari layer. Use "
    "this before analysis when the user refers to a channel by color instead of an "
    "exact layer name.",
    phase="1.5",
)
def resolve_channel(query: str) -> dict[str, Any]:
    layer_name = resolve_layer_name(query)
    return {"query": query, "layer": layer_name, "color": canonical_channel_color(query)}


@tool(
    name="resolve_target_channel",
    description="Resolve the target channel for cell-analysis workflows. Pass a layer "
    "name, color phrase (green/red/UV/IR), or marker (GFP/DAPI/...). Leave empty to use "
    "a confirmed target annotation, or — if only one image layer exists — assume that "
    "single layer. Counterstain channels are never auto-selected. Returns the resolved "
    "layer name and how it was resolved (explicit, annotation, phrase, inference).",
    phase="2",
)
def resolve_target_channel_tool(query: str | None = None) -> dict[str, Any]:
    try:
        result = resolve_target_channel(query)
    except AmbiguousChannelError as e:
        return {
            "ok": False,
            "error": str(e),
            "candidates": list(e.candidates),
        }
    return {
        "ok": True,
        "layer": result.layer,
        "source": result.source,
        "color": result.color,
        "note": result.note,
    }
