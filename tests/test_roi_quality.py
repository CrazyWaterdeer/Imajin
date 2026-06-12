"""ROI-quality scorer tests.

Starts as the v1 back-compat characterization (A0 of the v2.1 plan): pin the
current `roi_score` / `roi_confidence` contract so redefining `roi_confidence`
as v2.1 cannot silently regress the runner vision gate or downstream callers.
The v2.1 layers (routing / size / distribution / confidence) are added below as
they land.
"""

from __future__ import annotations

import numpy as np

from imajin.analysis.segmentation_auto3d import confidence_from_score
from imajin.tools import segment


# --- A0: v1 contract (must stay true through the v2.1 migration) ---


def test_v1_confidence_from_score_tiers_are_stable() -> None:
    good = {"n_objects": 12, "mask_fraction": 0.1, "largest_to_median_object_ratio": 2.0}
    # high requires score >= 75 and no critical (zero / region-level) warning
    assert confidence_from_score(90.0, good) == "high"
    assert confidence_from_score(74.9, good) == "medium"
    assert confidence_from_score(55.0, good) == "medium"
    assert confidence_from_score(54.9, good) == "low"
    # zero objects and region-level merges are always low regardless of score
    assert confidence_from_score(95.0, {"n_objects": 0}) == "low"
    region = {"n_objects": 3, "mask_fraction": 0.2, "largest_to_median_object_ratio": 30.0}
    assert confidence_from_score(95.0, region) == "low"


def test_v1_segment_target_objects_exposes_roi_contract(viewer) -> None:
    # Clean two-object field: the gate must stay quiet (high/medium, not low),
    # and the v1 fields the runner + tests depend on must be present.
    yy, xx = np.mgrid[:128, :128]
    image = (80.0 + xx * 0.15).astype(np.float32)
    image[28:40, 24:36] += 42.0
    image[88:102, 46:60] += 38.0
    viewer.add_image(image, name="target")

    res = segment.segment_target_objects(
        "target", background_radius=16, min_size=30, smoothing_sigma=0, fill_holes=False
    )

    assert isinstance(res["roi_score"], float)
    assert res["roi_confidence"] in {"high", "medium", "low"}
    assert res["roi_confidence"] != "low"  # a clean ROI must not read as a gross failure
    assert "qc_png_path" in res  # gate needs the overlay path


def test_v1_correct_roi_exposes_confidence_and_overlay(viewer) -> None:
    # correct_roi must surface roi_confidence + qc_png_path (H3) so the gate fires
    # on the correction itself — pin that contract before v2.1 changes confidence.
    image = np.zeros((128, 128), dtype=np.float32)
    image[30:50, 30:50] = 120.0
    viewer.add_image(image, name="img")
    seg = segment.segment_target_objects("img", background_radius=16, min_size=20, smoothing_sigma=0)

    res = segment.correct_roi("img", seg["labels_layer"], min_snr=3.0)
    assert res.get("ok") is True
    assert "roi_confidence" in res
    assert "qc_png_path" in res
