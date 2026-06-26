"""Tool-level tests for boundary_mask_from_shapes.

Real standalone ``napari.layers`` objects are injected into the (fake) viewer's
layer list, so the tool exercises real Shapes geometry + transforms without needing
``_FakeViewer.add_shapes`` (which does not exist).
"""
from __future__ import annotations

import numpy as np
import pytest

from imajin.tools import boundary, segment

napari_layers = pytest.importorskip("napari.layers")
Image = napari_layers.Image
Shapes = napari_layers.Shapes


def _rot(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _ellipse_corners(cy, cx, ry, rx, theta_deg) -> np.ndarray:
    own = np.array([[-ry, -rx], [-ry, rx], [ry, rx], [ry, -rx]], float)
    return (own @ _rot(theta_deg).T) + np.array([cy, cx])


def test_keeps_only_inside_and_wires_segmentation(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[20:35, 20:35] = 250.0  # inside the polygon
    img[70:85, 70:85] = 250.0  # outside the polygon
    viewer.layers.append(Image(img, name="img"))
    poly = np.array([[10, 10], [10, 50], [50, 50], [50, 10]], float)
    viewer.layers.append(Shapes([poly], shape_type=["polygon"], name="shp"))

    res = boundary.boundary_mask_from_shapes("shp", "img")
    assert res["ok"], res
    bmask = np.asarray(viewer.layers[res["boundary_layer"]].data)
    assert bmask[25, 25] > 0 and bmask[75, 75] == 0

    seg = segment.segment_target_objects(
        "img",
        boundary_mask=res["boundary_layer"],
        background_radius=0,
        smoothing_sigma=0,
        min_size=20,
        save_qc_png=False,
    )
    labels = np.asarray(viewer.layers[seg["labels_layer"]].data)
    assert (labels[20:35, 20:35] > 0).any(), "object inside boundary kept"
    assert (labels[70:85, 70:85] == 0).all(), "object outside boundary dropped"
    assert seg["threshold_scope"] == "boundary_mask"


def test_respects_scale_and_translate(viewer) -> None:
    # Image has non-unit scale + translate; the Shapes layer is at the default
    # (identity) transform, so its data coords are world coords. We draw the
    # rectangle at the *world* position of image-index rows/cols 40..60.
    sy = sx = 0.3
    ty, tx = 5.0, 2.0
    viewer.layers.append(
        Image(np.zeros((100, 100), np.float32), name="img", scale=(sy, sx), translate=(ty, tx))
    )
    y0, y1, x0, x1 = 40, 60, 40, 60
    corners = np.array(
        [
            [y0 * sy + ty, x0 * sx + tx],
            [y0 * sy + ty, x1 * sx + tx],
            [y1 * sy + ty, x1 * sx + tx],
            [y1 * sy + ty, x0 * sx + tx],
        ],
        float,
    )
    viewer.layers.append(Shapes([corners], shape_type=["rectangle"], name="shp"))

    res = boundary.boundary_mask_from_shapes("shp", "img")
    assert res["ok"], res
    out_layer = viewer.layers[res["boundary_layer"]]
    bmask = np.asarray(out_layer.data)
    ys, xs = np.where(bmask > 0)
    # The foreground must land on the intended image-index region, not an offset.
    assert abs(int(ys.min()) - y0) <= 1 and abs(int(ys.max()) - (y1 - 1)) <= 1
    assert abs(int(xs.min()) - x0) <= 1 and abs(int(xs.max()) - (x1 - 1)) <= 1
    assert tuple(float(v) for v in out_layer.translate) == (ty, tx)
    assert res["boundary_layer"] == out_layer.name


def test_ellipse_excludes_bbox_corners(viewer) -> None:
    viewer.layers.append(Image(np.zeros((60, 60), np.float32), name="img"))
    corners = _ellipse_corners(30, 30, 15, 5, 45)
    viewer.layers.append(Shapes([corners], shape_type=["ellipse"], name="e"))

    res = boundary.boundary_mask_from_shapes("e", "img")
    assert res["ok"], res
    m = np.asarray(viewer.layers[res["boundary_layer"]].data)
    assert m[30, 30] > 0
    for y, x in np.clip(np.round(corners).astype(int), 0, 59):
        assert m[y, x] == 0


def test_3d_reference_broadcasts_across_z(viewer) -> None:
    viewer.layers.append(Image(np.zeros((4, 50, 50), np.float32), name="img3d"))
    poly3d = np.array(
        [[1, 10, 10], [1, 10, 30], [1, 30, 30], [1, 30, 10]], float
    )  # drawn on z=1
    viewer.layers.append(Shapes([poly3d], shape_type=["polygon"], name="s3d"))

    res = boundary.boundary_mask_from_shapes("s3d", "img3d")
    assert res["ok"] and res["broadcast_z"] is True
    m = np.asarray(viewer.layers[res["boundary_layer"]].data)
    assert m.shape == (4, 50, 50)
    plane = m[1] > 0
    assert plane.any()
    for z in range(4):
        assert np.array_equal(m[z] > 0, plane), "every Z plane equals the drawn plane"
    assert (m[0] > 0).any(), "a non-drawn Z plane is also constrained"


def test_failure_modes(viewer) -> None:
    viewer.layers.append(Image(np.zeros((40, 40), np.float32), name="img"))
    viewer.layers.append(
        Shapes([np.array([[5, 5], [35, 35]], float)], shape_type=["line"], name="ln")
    )
    viewer.layers.append(Shapes(ndim=2, name="empty"))

    assert boundary.boundary_mask_from_shapes("ln", "img")["ok"] is False  # line only
    assert boundary.boundary_mask_from_shapes("empty", "img")["ok"] is False  # nothing drawn
    assert boundary.boundary_mask_from_shapes("img", "img")["ok"] is False  # non-Shapes
    assert boundary.boundary_mask_from_shapes("missing", "img")["ok"] is False  # bad name
