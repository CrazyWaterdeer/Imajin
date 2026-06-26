"""Pure-helper tests for boundary-mask rasterisation (no viewer/qapp).

Real ``napari.layers.Shapes`` objects are constructed standalone so the tests
exercise napari's actual vertex/shape_type representation, not hand-authored guesses.
"""
from __future__ import annotations

import numpy as np
import pytest

from imajin.tools.boundary import (
    broadcast_yx_to_ref,
    rasterize_shapes_yx,
)

napari_layers = pytest.importorskip("napari.layers")
Shapes = napari_layers.Shapes


def _rot(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _ellipse_corners(cy, cx, ry, rx, theta_deg) -> np.ndarray:
    own = np.array([[-ry, -rx], [-ry, rx], [ry, rx], [ry, -rx]], float)
    return (own @ _rot(theta_deg).T) + np.array([cy, cx])


def test_rotated_ellipse_is_filled_as_rotated_ellipse() -> None:
    # 45-degree, elongated ellipse (ry=15 >> rx=5): the long axis is the diagonal.
    cy, cx, ry, rx, theta = 30.0, 30.0, 15.0, 5.0, 45.0
    corners = _ellipse_corners(cy, cx, ry, rx, theta)
    s = Shapes([corners], shape_type=["ellipse"])  # real napari ellipse data
    mask = rasterize_shapes_yx(list(s.data), [str(t) for t in s.shape_type], (60, 60))

    assert mask.shape == (60, 60)
    assert mask.dtype == bool
    assert bool(mask[int(cy), int(cx)]), "ellipse centre must be foreground"

    # Bbox corners are outside any ellipse.
    for y, x in np.clip(np.round(corners).astype(int), 0, 59):
        assert not bool(mask[y, x]), "bbox corner must be background"

    # Rotation-sensitive: a step along the *rotated* major axis is inside; the same
    # step along the image y-axis is outside (it would be inside for an unrotated
    # ry=15 ellipse). This is what makes the test fail if rotation were dropped.
    d = 10.0
    major = _rot(theta) @ np.array([d, 0.0])  # rotated long-axis direction * d
    py, px = int(round(cy + major[0])), int(round(cx + major[1]))
    assert bool(mask[py, px]), "point along rotated major axis must be foreground"
    assert not bool(mask[int(cy + d), int(cx)]), (
        "point along image y must be background for a 45deg-rotated ellipse"
    )

    bbox_area = (2 * ry) * (2 * rx)
    assert mask.sum() < bbox_area, "an ellipse fills less than its bounding box"


def test_rectangle_and_polygon_fill_interior() -> None:
    rect = Shapes(
        [np.array([[10, 10], [10, 40], [40, 40], [40, 10]], float)],
        shape_type=["rectangle"],
    )
    m = rasterize_shapes_yx(list(rect.data), ["rectangle"], (50, 50))
    assert bool(m[25, 25]) and not bool(m[5, 5])

    tri = Shapes(
        [np.array([[5, 5], [5, 45], [45, 25]], float)], shape_type=["polygon"]
    )
    mp = rasterize_shapes_yx(list(tri.data), ["polygon"], (50, 50))
    assert bool(mp[10, 25]) and not bool(mp[44, 5])


def test_pair_filtering_keeps_type_alignment() -> None:
    # A dropped (None) shape in the MIDDLE: if types were not filtered together,
    # napari would get 2 polys + 3 types and raise. Result must still contain both
    # good shapes with their own geometry.
    good_a = np.array([[2, 2], [2, 18], [18, 18], [18, 2]], float)  # top-left square
    good_b = np.array([[60, 60], [60, 78], [78, 78], [78, 60]], float)  # bot-right
    mask = rasterize_shapes_yx(
        [good_a, None, good_b],
        ["polygon", "ellipse", "rectangle"],
        (80, 80),
    )
    assert bool(mask[10, 10]), "first kept shape present"
    assert bool(mask[70, 70]), "third kept shape present"


def test_empty_and_all_dropped_give_blank_mask() -> None:
    assert rasterize_shapes_yx([], [], (20, 20)).sum() == 0
    assert rasterize_shapes_yx([None], ["polygon"], (20, 20)).sum() == 0


def test_out_of_bounds_polygon_is_clipped() -> None:
    # Vertices spilling past the grid: napari clips to labels_shape, no index error.
    poly = np.array([[-10, -10], [-10, 30], [30, 30], [30, -10]], float)
    m = rasterize_shapes_yx([poly], ["polygon"], (20, 20))
    assert m.shape == (20, 20)
    assert bool(m[5, 5]) and m.sum() <= 20 * 20


def test_broadcast_yx_to_ref() -> None:
    mask2d = np.zeros((10, 12), dtype=bool)
    mask2d[2:5, 3:7] = True

    out2d, b2 = broadcast_yx_to_ref(mask2d, (10, 12))
    assert b2 is False and out2d.dtype == np.int32 and out2d.shape == (10, 12)

    out3d, b3 = broadcast_yx_to_ref(mask2d, (4, 10, 12))
    assert b3 is True and out3d.shape == (4, 10, 12) and out3d.dtype == np.int32
    for z in range(4):
        assert np.array_equal(out3d[z] > 0, mask2d), "every Z plane equals the 2D mask"
