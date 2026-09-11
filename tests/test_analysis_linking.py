"""Tests for the one candidate-scoring primitive genuinely shared between the
Z-stack plane linker (``segmentation_auto3d.stitch_plane_labels``) and the
time-axis linker (``roi_redetect.link_nearest``): :func:`area_ratio`. See both
modules' docstrings for the two structural reasons the rest of the two linkers
is deliberately NOT shared (union-find merge vs. share-an-id; overlap-first
gating breaking under drift). This file pins the one thing that genuinely is
shared -- both that its own arithmetic is correct, and that it really is ONE
function imported by both call sites, not two copies that could quietly drift
apart.
"""

from __future__ import annotations

import numpy as np

from imajin.analysis import roi_redetect, segmentation_auto3d
from imajin.analysis.segmentation_auto3d import _centroid_links, _overlap_links, area_ratio


def test_area_ratio_is_one_for_equal_areas() -> None:
    assert area_ratio(10, 10) == 1.0


def test_area_ratio_is_symmetric_and_reports_the_larger_over_the_smaller() -> None:
    assert area_ratio(10, 30) == 3.0
    assert area_ratio(30, 10) == 3.0


def test_area_ratio_floors_a_zero_area_at_one_pixel() -> None:
    # max(1, min(0, 5)) == 1 -- matches the pre-extraction inline expression
    # exactly (both original call sites already floored their own areas at 1
    # before computing this ratio); a genuine zero-pixel region should never
    # reach here in practice (regionprops never emits one), but the floor
    # keeps this a total function rather than a division by zero.
    assert area_ratio(0, 5) == 5.0


def test_roi_redetect_reuses_the_exact_same_function_not_a_second_copy() -> None:
    """Guards against a future contributor 'helpfully' re-inlining the ratio
    expression inside roi_redetect.py instead of importing this one -- the
    whole point of the extraction (see both modules' docstrings) is that
    there is exactly ONE area_ratio, not two copies that could drift apart."""
    assert roi_redetect.area_ratio is segmentation_auto3d.area_ratio


def test_overlap_links_rejects_a_pair_whose_areas_are_too_different() -> None:
    """A plane-to-plane pair with total (100%) overlap but a size ratio past
    the gate must not link -- exercised directly against the private
    candidate generator, since neither of stitch_plane_labels' two pinned
    edge/ambiguity tests (test_tools_segment.py) happens to stress this
    particular gate on its own."""
    current = np.zeros((10, 10), dtype=np.int32)
    current[1:9, 1:9] = 1  # 64px
    nxt = np.zeros((10, 10), dtype=np.int32)
    nxt[3:6, 3:6] = 1  # 9px, fully inside current's footprint -> 100% overlap fraction

    areas = {(0, 1): int((current == 1).sum()), (1, 1): int((nxt == 1).sum())}
    links = _overlap_links(current, nxt, areas, min_overlap_fraction=0.2, max_area_ratio=3.0, z=0)

    assert links == []  # overlap alone (100%) is not enough; 64/9 ~= 7.1x fails the gate


def test_overlap_links_allows_a_pair_within_the_area_gate() -> None:
    current = np.zeros((10, 10), dtype=np.int32)
    current[1:9, 1:9] = 1  # 64px
    nxt = np.zeros((10, 10), dtype=np.int32)
    nxt[1:9, 1:9] = 1  # same footprint -> full overlap, ratio 1.0

    areas = {(0, 1): 64, (1, 1): 64}
    links = _overlap_links(current, nxt, areas, min_overlap_fraction=0.2, max_area_ratio=3.0, z=0)

    assert len(links) == 1
    assert links[0][:2] == ((0, 1), (1, 1))


def test_centroid_links_rejects_a_pair_whose_areas_are_too_different() -> None:
    areas = {(0, 1): 10, (1, 1): 400}  # 40x apart
    centroids = {(0, 1): (5.0, 5.0), (1, 1): (5.0, 6.0)}  # 1px apart -- well inside any distance gate
    links = _centroid_links(
        [(0, 1)], [(1, 1)], areas, centroids, max_centroid_distance=10.0, max_area_ratio=3.0
    )
    assert links == []
