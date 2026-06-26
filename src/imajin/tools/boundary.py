"""Convert hand-drawn napari Shapes into a boundary mask for segmentation.

A user draws a polygon/rectangle/ellipse on the image and wants segmentation to
find ROIs only inside it. The segmentation tools already accept a ``boundary_mask``
*Labels* layer (see ``target_pipeline.threshold_and_label``); this module bridges a
*Shapes* layer to that layer.

Rasterisation uses napari's own ``Shapes.to_labels`` so the filled region matches
exactly what the user drew, including rotated ellipses (hand-rolled ellipse maths
would mis-handle rotation). Vertices are converted shapes-data -> world ->
reference-data so a Shapes layer whose transform differs from the image (the common
case: a fresh Shapes layer is identity-scaled while a micron-scaled image is not)
does not produce a silently offset mask. For a 3D ZYX reference the YX region is
broadcast across every Z plane.

napari is imported lazily (inside functions) so ``import imajin.tools`` stays
headless-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import layer_axes_from_metadata
from imajin.session import get_layer, get_viewer
from imajin.tools.registry import tool

_AREA = {"polygon", "rectangle", "ellipse"}
_SKIP = {"line", "path"}


def _clean_shape(vertices: object) -> np.ndarray | None:
    """Validate a whole shape's vertices.

    Returns the ``(N, D)`` float array, or ``None`` to drop the shape. A shape with
    any non-finite coordinate is dropped *entirely* rather than having rows removed,
    because deleting a row from a rectangle/ellipse's control points would corrupt
    its geometry.
    """
    a = np.asarray(vertices, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] < 3 or not np.isfinite(a).all():
        return None
    return a


def rasterize_shapes_yx(
    yx_polys: list[np.ndarray | None],
    area_types: list[str],
    target_yx: tuple[int, int],
) -> np.ndarray:
    """Union boolean mask of area shapes, rasterised by napari into ``target_yx``.

    ``yx_polys`` are already in the reference's YX index space. ``yx_polys`` and
    ``area_types`` are filtered **together** so dropping a degenerate shape can never
    misalign the remaining shapes' types. Empty input -> all-False mask.
    """
    from napari.layers import Shapes  # lazy: keep module import headless-safe

    Y, X = int(target_yx[0]), int(target_yx[1])
    pairs = [
        (p, t)
        for p, t in zip(yx_polys, area_types)
        if p is not None and np.asarray(p).shape[0] >= 3
    ]
    if not pairs:
        return np.zeros((Y, X), dtype=bool)
    polys, types = zip(*pairs)
    labels = Shapes(list(polys), shape_type=list(types)).to_labels(labels_shape=(Y, X))
    return np.asarray(labels) > 0


def broadcast_yx_to_ref(
    mask2d: np.ndarray, ref_shape: tuple[int, ...]
) -> tuple[np.ndarray, bool]:
    """Materialise a YX mask to match ``ref_shape`` as int32.

    For a 3D ``(Z, Y, X)`` reference the YX mask is broadcast across every Z plane.
    Returns ``(mask, broadcast_z)``.
    """
    mask2d = np.asarray(mask2d)
    if len(ref_shape) == 3:
        out = np.broadcast_to(mask2d[None], (int(ref_shape[0]), *mask2d.shape))
        return out.astype(np.int32), True
    return mask2d.astype(np.int32), False


def _apply_transform(tf: Any, coords: np.ndarray) -> np.ndarray:
    """Apply a napari layer transform to ``(N, D)`` coords, row-wise if it does not
    vectorise over the whole array."""
    coords = np.asarray(coords, dtype=float)
    try:
        out = np.asarray(tf(coords), dtype=float)
        if out.shape == coords.shape:
            return out
    except (TypeError, ValueError, IndexError):
        pass
    return np.array([np.asarray(tf(c), dtype=float) for c in coords])


def _build_boundary_mask_from_layers(
    shapes: Any, reference: Any, name: str | None = None
) -> dict[str, Any]:
    """Core (viewer-free): rasterise ``shapes`` into a mask matching ``reference``.

    Takes live layer objects so it is unit-testable with standalone napari layers.
    On success returns ``{"ok": True, "mask": <int32 array>, ...}``; the caller adds
    the labels layer. On any rejected input returns ``{"ok": False, "error": ...}``.
    """
    import napari.layers  # lazy

    if not isinstance(shapes, napari.layers.Shapes):
        return {
            "ok": False,
            "error": f"shapes_layer must be a Shapes layer, got {type(shapes).__name__}",
        }

    data = getattr(reference, "data", None)
    ndim = getattr(data, "ndim", None)
    shape = getattr(data, "shape", None)
    if ndim is None or shape is None or isinstance(data, (list, tuple)):
        return {
            "ok": False,
            "error": "reference_layer has no array data (multiscale or non-image "
            "layers are unsupported)",
        }

    axes = layer_axes_from_metadata(getattr(reference, "metadata", {}) or {}, int(ndim))
    if axes not in ("YX", "ZYX"):
        return {
            "ok": False,
            "error": f"reference_layer axes {axes!r} unsupported; need a 2D YX image "
            "or 3D ZYX stack (extract a timepoint/slice first)",
        }

    shape_types = [str(t).lower() for t in shapes.shape_type]
    raw = list(shapes.data)
    n_shapes = len(raw)
    skipped: dict[str, int] = {}
    yx_list: list[np.ndarray] = []
    types: list[str] = []
    for verts, t in zip(raw, shape_types):
        if t in _SKIP:
            skipped[t] = skipped.get(t, 0) + 1
            continue
        if t not in _AREA:
            return {
                "ok": False,
                "error": f"unsupported shape_type {t!r} (expected polygon/rectangle/"
                "ellipse, or line/path which are skipped)",
            }
        cleaned = _clean_shape(verts)
        if cleaned is None:
            continue
        if cleaned.shape[1] != int(ndim):
            return {
                "ok": False,
                "error": f"shape has {cleaned.shape[1]}-D vertices but reference is "
                f"{int(ndim)}-D; draw the shape on this image (a 3D shape carries its "
                "Z coordinate), or extract a 2D slice first",
            }
        world = _apply_transform(shapes.data_to_world, cleaned)
        refc = _apply_transform(reference.world_to_data, world)
        yx_list.append(np.asarray(refc)[:, -2:])
        types.append(t)

    target_yx = (int(shape[-2]), int(shape[-1]))
    mask2d = rasterize_shapes_yx(yx_list, types, target_yx)
    mask, broadcast_z = broadcast_yx_to_ref(mask2d, tuple(int(s) for s in shape))

    voxels = int(mask.sum())
    if voxels == 0:
        return {
            "ok": False,
            "error": "boundary mask is empty (no area shapes, all off-image, or "
            "degenerate vertices)",
            "n_shapes": int(n_shapes),
            "skipped_shape_types": skipped,
        }

    warnings: list[str] = []
    if broadcast_z:
        warnings.append(
            f"shape drawn on one plane was broadcast across all {int(shape[0])} Z planes"
        )
    if voxels / mask.size > 0.98:
        warnings.append("boundary covers ~the entire frame; check the drawing")

    return {
        "ok": True,
        "mask": mask,
        "n_shapes": int(n_shapes),
        "n_used": int(len(types)),
        "skipped_shape_types": skipped,
        "mask_voxels": voxels,
        "mask_fraction": float(voxels / mask.size),
        "axes": axes,
        "broadcast_z": bool(broadcast_z),
        "warnings": warnings,
    }


@tool(
    description="Convert a hand-drawn napari Shapes layer (polygon/rectangle/ellipse) "
    "into a boundary Labels mask matching a reference image, so segmentation can be "
    "constrained to inside the drawn region by passing the result as boundary_mask. "
    "Open shapes (line/path) are ignored. For a 3D ZYX reference the region is "
    "broadcast across every Z plane. Draw the shape on the same image you will segment.",
    phase="2",
    llm=True,
    worker=False,
)
def boundary_mask_from_shapes(
    shapes_layer: str,
    reference_layer: str,
    name: str | None = None,
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        try:
            shapes = get_layer(shapes_layer)
        except KeyError:
            return {"ok": False, "error": f"shapes_layer {shapes_layer!r} not found"}
        try:
            reference = get_layer(reference_layer)
        except KeyError:
            return {
                "ok": False,
                "error": f"reference_layer {reference_layer!r} not found",
            }

        info = _build_boundary_mask_from_layers(shapes, reference, name)
        if not info.get("ok"):
            return info

        mask = info.pop("mask")
        out_name = name or f"{reference_layer}_boundary"
        scale_attr = getattr(reference, "scale", None)
        scale = (
            tuple(float(s) for s in scale_attr)
            if scale_attr is not None and len(scale_attr) > 0
            else None
        )
        layer = get_viewer().add_labels(
            mask,
            name=out_name,
            scale=scale or None,
            metadata={
                "source_shapes_layer": shapes_layer,
                "reference_layer": reference_layer,
                "axes": info["axes"],
                "broadcast_z": info["broadcast_z"],
                "mask_voxels": info["mask_voxels"],
                "mask_fraction": info["mask_fraction"],
                "skipped_shape_types": info["skipped_shape_types"],
            },
        )
        translate = getattr(reference, "translate", None)
        if translate is not None:
            try:
                layer.translate = translate
            except Exception:  # noqa: BLE001 - some viewers/layers lack translate
                pass
        # Return the layer's *actual* name (napari auto-renames a duplicate to
        # "name [1]") so a follow-up segment call targets the right layer.
        info["boundary_layer"] = layer.name
        return info

    return call_on_main(_run)
