from __future__ import annotations

import numpy as np

from imajin.analysis.segmentation import dilate_binary_um, erode_binary_um


def _filled_square(field: int, lo: int, hi: int) -> np.ndarray:
    m = np.zeros((field, field), dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


def test_erode_shrinks_square_by_radius_isotropic() -> None:
    # 10x10 solid, radius 2px each side -> 6x6.
    m = _filled_square(20, 5, 15)
    out = erode_binary_um(m, spacing=(1.0, 1.0), radius_um=2.0)
    assert int(out.sum()) == 6 * 6


def test_erode_uses_per_axis_radius_when_anisotropic() -> None:
    # spacing (y=2um, x=1um), radius 2um -> y radius 1px, x radius 2px.
    m = _filled_square(20, 5, 15)  # 10x10
    out = erode_binary_um(m, spacing=(2.0, 1.0), radius_um=2.0)
    assert int(out.sum()) == (10 - 2) * (10 - 4)  # 8 * 6


def test_erode_noop_for_zero_or_negative_radius() -> None:
    m = _filled_square(20, 5, 15)
    assert np.array_equal(erode_binary_um(m, spacing=(1.0, 1.0), radius_um=0.0), m)
    assert np.array_equal(erode_binary_um(m, spacing=(1.0, 1.0), radius_um=-3.0), m)


def test_erode_removes_thin_line() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[10, 2:18] = True  # 1px-wide line
    out = erode_binary_um(m, spacing=(1.0, 1.0), radius_um=1.0)
    assert int(out.sum()) == 0


def test_closing_is_identity_on_solid_blob_with_margin() -> None:
    # dilate then erode by the same radius returns a solid convex blob unchanged
    # when there is >= radius margin to every array edge.
    m = _filled_square(20, 6, 14)  # 8x8, margin 6
    dil = dilate_binary_um(m, spacing=(1.0, 1.0), radius_um=2.0)
    closed = erode_binary_um(dil, spacing=(1.0, 1.0), radius_um=2.0)
    assert np.array_equal(closed, m)


def test_erode_3d_shrinks_all_axes() -> None:
    m = np.zeros((12, 12, 12), dtype=bool)
    m[3:9, 3:9, 3:9] = True  # 6^3 solid
    out = erode_binary_um(m, spacing=(1.0, 1.0, 1.0), radius_um=1.0)
    assert int(out.sum()) == 4 ** 3
