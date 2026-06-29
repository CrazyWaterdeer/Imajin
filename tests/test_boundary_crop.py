"""Pure tests for ROI bbox-crop helpers (no viewer)."""
from __future__ import annotations

import numpy as np

from imajin.analysis.segmentation import boundary_bbox_slices, scatter_labels_to_full


def test_bbox_slices_offcentre_with_margin_clips_to_bounds() -> None:
    m = np.zeros((100, 100), dtype=bool)
    m[20:40, 60:75] = True  # ys 20..39, xs 60..74
    sl = boundary_bbox_slices(m, (100, 100), margin=10)
    assert sl is not None
    ys, xs = sl
    assert (ys.start, ys.stop) == (10, 50)  # 20-10 .. 39+1+10
    assert (xs.start, xs.stop) == (50, 85)  # 60-10 .. 74+1+10
    # margin that would overflow is clipped to image bounds
    sl2 = boundary_bbox_slices(m, (100, 100), margin=1000)
    assert sl2 == (slice(0, 100), slice(0, 100)) or sl2 is None
    # ^ a margin spanning the whole frame -> None (no benefit)
    assert boundary_bbox_slices(m, (100, 100), margin=1000) is None


def test_bbox_slices_empty_and_wholeframe_return_none() -> None:
    assert boundary_bbox_slices(np.zeros((50, 50), bool), (50, 50), 5) is None
    full = np.ones((50, 50), dtype=bool)
    assert boundary_bbox_slices(full, (50, 50), 0) is None


def test_bbox_slices_3d_raw_keeps_full_z() -> None:
    m = np.zeros((64, 64), dtype=bool)
    m[10:20, 10:20] = True
    sl = boundary_bbox_slices(m, (8, 64, 64), margin=4)
    assert sl is not None and len(sl) == 3
    assert sl[0] == slice(None)  # full Z
    assert sl[1] == slice(6, 24) and sl[2] == slice(6, 24)


def test_scatter_labels_places_crop_and_is_int32() -> None:
    crop = np.array([[0, 1], [2, 2]], dtype=np.int64)
    sl = (slice(3, 5), slice(7, 9))
    out = scatter_labels_to_full(crop, (10, 12), sl)
    assert out.shape == (10, 12) and out.dtype == np.int32
    assert np.array_equal(out[3:5, 7:9], crop)
    out_zeroed = out.copy()
    out_zeroed[3:5, 7:9] = 0
    assert not out_zeroed.any(), "no labels outside the scatter window"
    assert set(np.unique(out)) == {0, 1, 2}
