from __future__ import annotations

from typing import Any

from imajin.session import (
    AmbiguousChannelError,
    canonical_channel_color,
    canonical_channel_role,
    get_layer,
    get_sample,
    list_channel_annotations,
    put_channel_annotation,
    resolve_layer_name,
    resolve_target_channel,
    viewer_or_none,
)
from imajin.tools.registry import tool


_NUCLEAR_MARKERS: dict[str, bool] = {
    "topro": True,
    "to-pro": True,
    "to pro": True,
    "to-pro-3": True,
    "topro-3": True,
    "topro3": True,
    "dapi": True,
    "hoechst": True,
    "draq5": True,
    "nc82": False,
    "bruchpilot": False,
    "phalloidin": False,
}


def _normalize_marker(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower().replace("_", "-")


def _marker_is_nuclear(marker: str | None) -> bool | None:
    norm = _normalize_marker(marker)
    if norm is None:
        return None
    return _NUCLEAR_MARKERS.get(norm)


def _layer_name_suggests_far_red(layer_name: str) -> bool:
    text = layer_name.lower().replace("_", " ")
    keywords = ("633", "640", "647", "far red", "farred")
    return any(k in text for k in keywords)


@tool(
    description="Annotate an image layer's channel identity. Keep this simple: role is "
    "target, counterstain, ignore, or unknown. Color understands green, red, UV, "
    "and IR/far red aliases. Only a confirmed target channel is the default for "
    "segmentation, intensity, cell size, and time-course measurement.",
    phase="1.5",
)
def annotate_channel(
    layer: str,
    role: str = "unknown",
    color: str | None = None,
    marker: str | None = None,
    biological_target: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    resolved_role = canonical_channel_role(role)
    layer_name = put_channel_annotation(
        layer_name=layer,
        role=role,
        color=color,
        marker=marker,
        biological_target=biological_target,
        notes=notes,
    )
    if resolved_role == "counterstain":
        try:

            L = get_layer(layer_name)
            if hasattr(L, "colormap"):
                L.colormap = "gray"
        except Exception:
            pass
    return {
        "layer": layer_name,
        "role": resolved_role,
        "color": canonical_channel_color(color),
        "marker": marker,
        "biological_target": biological_target,
        "notes": notes,
    }


@tool(
    name="list_channel_annotations",
    description="List current channel annotations: target/counterstain/ignore/unknown "
    "role, canonical color (green/red/uv/ir), marker, and biological target.",
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


@tool(
    name="detect_counterstain_channel",
    description="Identify the counterstain channel for the current sample (or all "
    "loaded layers if no sample given). Resolution priority: layers annotated as "
    "role=counterstain first; otherwise layers whose name suggests a far-red "
    "(633/640/647) wavelength. Returns confidence and whether the marker is "
    "nuclear (TOPRO/DAPI/Hoechst). Used by expression-domain workflows to decide "
    "whether to intersect the reporter mask with a structural counterstain.",
    phase="1.5",
)
def detect_counterstain_channel(
    sample_name: str | None = None,
) -> dict[str, Any]:

    sample_layer_names: set[str] | None = None
    if sample_name is not None:
        sample = get_sample(sample_name)
        sample_layer_names = {str(n) for n in (sample.layers or [])}

    annotations = list_channel_annotations()
    annotated_counterstain = [
        entry
        for entry in annotations
        if entry.get("role") == "counterstain"
        and (
            sample_layer_names is None
            or entry.get("layer_name") in sample_layer_names
        )
    ]
    if annotated_counterstain:
        first = annotated_counterstain[0]
        marker = first.get("marker")
        return {
            "counterstain_layer": first.get("layer_name"),
            "counterstain_marker": _normalize_marker(marker),
            "is_nuclear": _marker_is_nuclear(marker),
            "confidence": "annotated",
            "needs_user_confirmation": False,
            "candidate_layers": [
                entry.get("layer_name") for entry in annotated_counterstain
            ],
        }

    viewer = viewer_or_none()
    candidates: list[str] = []
    if viewer is not None:
        for layer in viewer.layers:
            name = str(layer.name)
            if sample_layer_names is not None and name not in sample_layer_names:
                continue
            if _layer_name_suggests_far_red(name):
                candidates.append(name)

    if candidates:
        return {
            "counterstain_layer": candidates[0],
            "counterstain_marker": None,
            "is_nuclear": None,
            "confidence": "inferred",
            "needs_user_confirmation": True,
            "candidate_layers": candidates,
        }

    return {
        "counterstain_layer": None,
        "counterstain_marker": None,
        "is_nuclear": None,
        "confidence": "none",
        "needs_user_confirmation": False,
        "candidate_layers": [],
    }
