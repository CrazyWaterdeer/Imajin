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

import numpy as np

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
