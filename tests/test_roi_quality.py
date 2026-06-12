"""ROI-quality scorer tests.

Starts as the v1 back-compat characterization (A0 of the v2.1 plan): pin the
current `roi_score` / `roi_confidence` contract so redefining `roi_confidence`
as v2.1 cannot silently regress the runner vision gate or downstream callers.
The v2.1 layers (routing / size / distribution / confidence) are added below as
they land.
"""

from __future__ import annotations

import numpy as np

from imajin.analysis.roi_quality import (
    correction_materiality,
    distribution_flag,
    effective_object_count,
    object_class,
    object_sizes_physical,
    roi_confidence_v2,
    route,
)
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


# --- A: Layer-0 routing ---


def test_object_class_from_metadata() -> None:
    assert object_class({"segmentation_method": "target_objects"}) == "blob"
    assert object_class({"segmentation_method": "auto_target_objects"}) == "blob"
    assert object_class({"segmentation_method": "expression_domain"}) == "domain"
    assert object_class({"segmentation_method": "neuron_trace"}) == "neuron"
    assert object_class({"object_unit": "object_or_roi"}) == "blob"  # fallback
    assert object_class({"object_unit": "nucleus"}) == "blob"
    assert object_class({}) == "unclassified"
    assert object_class(None) == "unclassified"


def test_effective_object_count_distinguishes_spread_from_clustered() -> None:
    # 16 objects on a wide grid -> distributed
    spread = np.zeros((128, 128), dtype=np.int32)
    label = 1
    for y in range(10, 120, 30):
        for x in range(10, 120, 30):
            spread[y : y + 4, x : x + 4] = label
            label += 1
    n, distributed = effective_object_count(spread)
    assert n >= 12 and distributed is True

    # same count crammed into one corner -> not distributed
    clustered = np.zeros((128, 128), dtype=np.int32)
    label = 1
    for y in range(2, 26, 6):
        for x in range(2, 26, 6):
            clustered[y : y + 2, x : x + 2] = label
            label += 1
    n2, distributed2 = effective_object_count(clustered)
    assert n2 >= 12 and distributed2 is False

    # too few objects -> never "distributed"
    assert effective_object_count(np.zeros((16, 16), dtype=np.int32)) == (0, False)


def test_route_admits_distribution_only_for_numerous_distributed_blobs() -> None:
    assert "distribution" in route("blob", 12, True)
    assert "distribution" not in route("blob", 4, True)  # too few
    assert "distribution" not in route("blob", 12, False)  # clustered
    assert "distribution" not in route("domain", 50, True)  # domain never
    assert "distribution" not in route("neuron", 50, True)  # arbor never
    assert "distribution" not in route("unclassified", 50, True)  # conservative
    # structural + vision are always available
    for cls in ("blob", "domain", "neuron", "unclassified"):
        assert {"structural", "vision"} <= route(cls, 50, True)


# --- B: physical size extraction ---


def test_object_sizes_physical_2d_area_with_spacing() -> None:
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[20:30, 20:30] = 1  # 10x10 = 100 px, away from the border
    sizes, border_mask, n_usable = object_sizes_physical(labels, spacing=(0.5, 0.5))
    assert sizes.shape == (1,)
    assert np.isclose(sizes[0], 100 * 0.25)  # 25 µm²
    assert border_mask.tolist() == [False]
    assert n_usable == 1


def test_object_sizes_physical_3d_volume_anisotropic() -> None:
    labels = np.zeros((6, 32, 32), dtype=np.int32)
    labels[1:4, 10:14, 10:14] = 1  # 3*4*4 = 48 voxels, interior
    sizes, _border, _n = object_sizes_physical(labels, spacing=(2.0, 0.5, 0.5))
    assert np.isclose(sizes[0], 48 * (2.0 * 0.5 * 0.5))  # 24 µm³


def test_object_sizes_physical_flags_border_objects() -> None:
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[0:5, 0:5] = 1  # touches the top-left corner -> border
    labels[15:20, 15:20] = 2  # interior
    sizes, border_mask, n_usable = object_sizes_physical(labels)
    assert border_mask.tolist() == [True, False]
    assert n_usable == 1  # only the interior object is usable
    assert sizes.shape == (2,)  # all sizes returned; caller masks


# --- F0: synthetic size-distribution generators (calibration harness) ---
# Reused by the distribution-flag tests (C) and the end-to-end validation (F1).


def good_unimodal_sizes(n=50, base=20.0, cv=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return base * (1.0 + cv * rng.standard_normal(n))


def undersegmented_sizes(n=50, base=20.0, seed=0):
    # half the objects are merged doublets (~2x) -> bimodal at +1.0 log2
    rng = np.random.default_rng(seed)
    singles = base * (1.0 + 0.05 * rng.standard_normal(n // 2))
    doublets = 2.0 * base * (1.0 + 0.05 * rng.standard_normal(n - n // 2))
    return np.concatenate([singles, doublets])


def oversegmented_sizes(n=52, base=20.0, frac=0.25, seed=0):
    # a fragment tail well below the median
    rng = np.random.default_rng(seed)
    k = int(round(n * frac))
    main = base * (1.0 + 0.05 * rng.standard_normal(n - k))
    frags = (base / 4.0) * (1.0 + 0.05 * rng.standard_normal(k))
    return np.concatenate([main, frags])


def broad_lognormal_sizes(n=50, base=20.0, sigma=0.4, seed=0):
    # wide but *unimodal* biological spread — must NOT be flagged as error
    rng = np.random.default_rng(seed)
    return base * np.exp(sigma * rng.standard_normal(n))


def test_f0_generators_have_their_intended_shape() -> None:
    # Validates the harness the C / F1 tests rely on.
    import numpy as np

    under = undersegmented_sizes()
    # two clusters ~2x apart
    lo = under[under < 1.5 * 20.0]
    hi = under[under >= 1.5 * 20.0]
    assert len(lo) > 5 and len(hi) > 5

    over = oversegmented_sizes()
    assert np.mean(over < 20.0 / 2) > 0.2  # substantial small tail

    broad = broad_lognormal_sizes()
    assert broad.min() > 0 and broad.std() / broad.mean() > 0.2  # genuinely broad


# --- C: distribution anomaly flag (weak, medium-only) ---


def test_distribution_flag_catches_undersegmentation() -> None:
    res = distribution_flag(undersegmented_sizes(), n_eff=50)
    assert res["flag"] is True
    assert res["reason"] == "possible_undersegmentation"
    assert res["abstained"] is False


def test_distribution_flag_catches_oversegmentation() -> None:
    res = distribution_flag(oversegmented_sizes(), n_eff=52)
    assert res["flag"] is True
    assert res["reason"] == "possible_oversegmentation"


def test_distribution_flag_quiet_on_clean_unimodal() -> None:
    res = distribution_flag(good_unimodal_sizes(), n_eff=50)
    assert res["flag"] is False
    assert res["abstained"] is False


def test_distribution_flag_does_not_mistake_broad_biology_for_error() -> None:
    # The central safety property: wide *unimodal* lognormal spread (real
    # biology, e.g. lipid droplets under diet) must NOT be flagged.
    res = distribution_flag(broad_lognormal_sizes(), n_eff=50)
    assert res["flag"] is False


def test_distribution_flag_abstains_below_min_n() -> None:
    res = distribution_flag(good_unimodal_sizes(n=8), n_eff=8)
    assert res["abstained"] is True
    assert res["flag"] is False


def test_distribution_flag_is_only_ever_a_flag_never_a_confidence() -> None:
    # Spec-central: this layer never emits low/high and never a score.
    for sizes, n in [
        (undersegmented_sizes(), 50),
        (oversegmented_sizes(), 52),
        (good_unimodal_sizes(), 50),
        (good_unimodal_sizes(n=8), 8),
    ]:
        res = distribution_flag(sizes, n_eff=n)
        assert set(res) == {"flag", "reason", "metric", "abstained"}
        assert isinstance(res["flag"], bool)
        assert res["reason"] in {None, "possible_undersegmentation", "possible_oversegmentation"}


# --- D: confidence v2.1 mapping + correction materiality ---

_GOOD = {"n_objects": 12, "mask_fraction": 0.1, "largest_to_median_object_ratio": 2.0}
_CLEAN_FLAG = {"flag": False, "reason": None, "metric": None, "abstained": False}
_BLOB_ROUTE = {"structural", "vision", "distribution"}


def test_confidence_v2_high_needs_structure_and_clean_distribution() -> None:
    conf, drivers = roi_confidence_v2(
        90.0, _GOOD, route_layers=_BLOB_ROUTE, n_eff=12, obj_class="blob", dist_flag=_CLEAN_FLAG
    )
    assert conf == "high"
    assert drivers["driver"] == "structural_strong_and_distribution_clean"


def test_confidence_v2_domain_capped_at_medium_even_when_structurally_strong() -> None:
    # No distribution layer in the route -> cannot earn high (closes v1 hole).
    conf, _ = roi_confidence_v2(
        95.0, _GOOD, route_layers={"structural", "vision"}, n_eff=1, obj_class="domain",
        dist_flag=None,
    )
    assert conf == "medium"


def test_confidence_v2_distribution_flag_routes_to_medium_never_low() -> None:
    flag = {"flag": True, "reason": "possible_undersegmentation", "metric": 0.9, "abstained": False}
    conf, drivers = roi_confidence_v2(
        90.0, _GOOD, route_layers=_BLOB_ROUTE, n_eff=12, obj_class="blob", dist_flag=flag
    )
    assert conf == "medium"
    assert drivers["driver"] == "possible_undersegmentation"


def test_confidence_v2_gross_structural_failure_is_low() -> None:
    conf, _ = roi_confidence_v2(
        40.0, {"n_objects": 0}, route_layers=_BLOB_ROUTE, n_eff=0, obj_class="blob",
        dist_flag=None,
    )
    assert conf == "low"


def test_confidence_v2_abstained_distribution_cannot_be_high() -> None:
    abstained = {"flag": False, "reason": None, "metric": None, "abstained": True}
    conf, _ = roi_confidence_v2(
        90.0, _GOOD, route_layers=_BLOB_ROUTE, n_eff=8, obj_class="blob", dist_flag=abstained
    )
    assert conf == "medium"  # we never actually checked the distribution


def test_confidence_v2_material_correction_gap_is_medium() -> None:
    conf, drivers = roi_confidence_v2(
        90.0, _GOOD, route_layers=_BLOB_ROUTE, n_eff=12, obj_class="blob",
        dist_flag=_CLEAN_FLAG, correction_gap=True,
    )
    assert conf == "medium"
    assert drivers["driver"] == "correction_changed_measurement"


def test_correction_materiality() -> None:
    base = {"n_objects": 10, "object_area_median": 100.0}
    assert correction_materiality(base, dict(base)) is False
    assert correction_materiality(base, {"n_objects": 5, "object_area_median": 100.0}) is True
    assert correction_materiality(base, {"n_objects": 10, "object_area_median": 200.0}) is True
    assert correction_materiality({"n_objects": 0}, {"n_objects": 3}) is True
