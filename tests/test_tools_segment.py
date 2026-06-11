from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imajin.analysis.segmentation_auto3d import (
    SegmentationCandidate,
    confidence_from_score,
    filter_labels_by_z_extent,
    rank_segmentation_labels,
    score_roi_quality,
    selection_confidence,
    stitch_plane_labels,
)
from imajin.tools import segment


@pytest.mark.slow
def test_cellpose_sam_finds_blobs_2d(viewer, synthetic_blob_image) -> None:
    viewer.add_image(synthetic_blob_image, name="blobs")

    res = segment.cellpose_sam("blobs", do_3D=False, diameter=10)

    assert res["labels_layer"] == "blobs_masks"
    assert res["n_cells"] >= 3, f"expected >=3 cells, got {res['n_cells']}"
    labels = np.asarray(viewer.layers["blobs_masks"].data)
    assert labels.shape == synthetic_blob_image.shape
    assert labels.max() == res["n_cells"]


@pytest.mark.slow
def test_cellpose_sam_propagates_scale(viewer, synthetic_blob_image) -> None:
    viewer.add_image(synthetic_blob_image, name="blobs", scale=(0.2, 0.2))
    segment.cellpose_sam("blobs", do_3D=False, diameter=10)
    out = viewer.layers["blobs_masks"]
    assert tuple(float(s) for s in out.scale) == (0.2, 0.2)


def test_cellpose_sam_rejects_non_2d_3d(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(2, 4, 16, 16), dtype=np.uint16)
    viewer.add_image(data, name="huge", metadata={"axes": "ZYXA"})
    with pytest.raises(ValueError, match="2D \\(YX\\) or 3D \\(ZYX\\)"):
        segment.cellpose_sam("huge")


def test_cellpose_sam_rejects_time_series(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(4, 16, 16), dtype=np.uint16)
    viewer.add_image(data, name="movie", metadata={"axes": "TYX"})
    with pytest.raises(ValueError, match="time-series"):
        segment.cellpose_sam("movie")


def test_cellpose_sam_writes_qc_png(viewer, monkeypatch, tmp_path) -> None:
    image = np.zeros((32, 32), dtype=np.uint16)
    image[8:16, 8:16] = 1000
    labels = np.zeros_like(image, dtype=np.int32)
    labels[8:16, 8:16] = 1
    viewer.add_image(image, name="cell_image")

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return labels, None, None

    monkeypatch.setattr(segment, "_get_cellpose_model", lambda *_args, **_kwargs: _FakeModel())
    out = tmp_path / "segmentation_qc.png"

    res = segment.cellpose_sam(
        "cell_image",
        do_3D=False,
        qc_png_path=str(out),
    )

    assert res["labels_layer"] == "cell_image_masks"
    assert res["qc_png_path"] == str(out.resolve())
    assert res["qc_png_error"] is None
    assert out.exists()
    rgb = np.asarray(Image.open(out))
    assert rgb[12, 12, 0] != rgb[12, 12, 1], "ROI interior should be mask-filled"


def test_segmentation_qc_png_defaults_to_results_root(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    from pathlib import Path
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    reset_process_bundle()
    try:
        image = np.zeros((256, 256), dtype=np.float32)
        image[90:105, 100:115] = 100
        viewer.add_image(image, name="target")

        res = segment.segment_target_objects(
            "target",
            background_radius=16,
            min_size=20,
            smoothing_sigma=0,
        )

        # QC PNG must land inside the results root under a bundle's qc/ subdirectory.
        qc_path = Path(res["qc_png_path"])
        results_root = tmp_path / "results"
        assert str(qc_path).startswith(str(results_root)), (
            f"QC PNG {qc_path} is not under results root {results_root}"
        )
        assert qc_path.parent.name == "qc"
        assert Image.open(res["qc_png_path"]).mode == "RGB"
    finally:
        reset_process_bundle()


def test_segmentation_qc_png_skips_tiny_default_outputs(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:18, 8:18] = 100
    viewer.add_image(image, name="tiny_target")

    res = segment.segment_target_objects(
        "tiny_target",
        background_radius=8,
        min_size=10,
        smoothing_sigma=0,
    )

    assert res["qc_png_path"] is None
    assert res["qc_png_error"] is None
    assert "small image plane" in res["qc_png_skipped_reason"]
    out = tmp_path / "results" / "segmentation_qc"
    if out.exists():
        assert not list(out.glob("*.png"))


def test_segment_intensity_regions_selects_bright_foreground(viewer) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[5:30, 35:60] = 10.0  # dim tissue/background should not become an ROI
    image[10:20, 10:20] = 100.0
    image[38:50, 42:54] = 120.0
    viewer.add_image(image, name="reporter")

    res = segment.segment_intensity_regions(
        "reporter",
        threshold_method="percentile",
        percentile=90,
        min_size=20,
        smoothing_sigma=0,
        fill_holes=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["n_regions"] == 2
    assert labels[15, 15] > 0
    assert labels[44, 48] > 0
    assert labels[10, 40] == 0


def test_segment_intensity_regions_can_measure_cluster_as_one_roi(viewer) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:42, 18:45] = 100.0
    image[24:38, 45:54] = 90.0  # touching bright extension; keep as one cluster
    viewer.add_image(image, name="cluster")

    res = segment.segment_intensity_regions(
        "cluster",
        threshold_method="percentile",
        percentile=80,
        min_size=20,
        smoothing_sigma=0,
        split_touching=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["n_regions"] == 1
    assert labels[30, 30] == labels[30, 48] != 0


def test_segment_target_objects_uses_local_background(viewer) -> None:
    yy, xx = np.mgrid[:128, :128]
    image = (80.0 + xx * 0.15).astype(np.float32)
    image[15:78, 72:122] += 20.0  # broad high-gain background/tissue field
    image[28:40, 24:36] += 42.0
    image[88:102, 46:60] += 38.0
    viewer.add_image(image, name="target")

    res = segment.segment_target_objects(
        "target",
        background_radius=16,
        min_size=30,
        smoothing_sigma=0,
        fill_holes=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["n_objects"] == 2
    assert labels[34, 30] > 0
    assert labels[94, 52] > 0
    assert labels[30, 90] == 0
    assert res["top_bright_outside_fraction"] < 0.25
    # ROI-quality fields surfaced for the agent-vision gate (C0.4)
    assert isinstance(res["roi_score"], float)
    assert res["roi_confidence"] in {"high", "medium"}  # a clean 2-object ROI is not ambiguous


def test_segment_target_objects_treats_unannotated_3d_as_z_stack(viewer) -> None:
    image = np.zeros((4, 64, 64), dtype=np.float32)
    image[1:4, 22:34, 24:36] = 90.0
    viewer.add_image(image, name="stack")

    res = segment.segment_target_objects(
        "stack",
        background_radius=8,
        min_size=20,
        smoothing_sigma=0,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["axes"] == "ZYX"
    assert labels.shape == image.shape
    assert res["n_objects"] >= 1


def test_stitch_plane_labels_links_overlapping_rois() -> None:
    labels = np.zeros((3, 24, 24), dtype=np.int32)
    labels[0, 5:14, 5:14] = 1
    labels[1, 6:15, 6:15] = 1
    labels[2, 2:8, 16:22] = 1

    stitched, record = stitch_plane_labels(labels, min_overlap_fraction=0.2)

    assert record["plane_roi_count"] == 3
    assert record["stitched_cell_count"] == 2
    assert stitched[0, 8, 8] == stitched[1, 9, 9] != 0
    assert stitched[2, 4, 18] != stitched[0, 8, 8]


def test_stitch_plane_labels_uses_one_to_one_ambiguous_links() -> None:
    labels = np.zeros((2, 24, 24), dtype=np.int32)
    labels[0, 5:15, 5:15] = 1
    labels[1, 5:13, 5:13] = 1
    labels[1, 5:13, 13:21] = 2

    stitched, record = stitch_plane_labels(labels, min_overlap_fraction=0.1)

    assert record["ambiguous_stitch_pairs"] > 0
    assert record["stitched_cell_count"] == 2
    assert stitched[0, 8, 8] == stitched[1, 8, 8] != 0
    assert stitched[1, 8, 16] != stitched[0, 8, 8]


def test_filter_labels_by_z_extent_removes_single_plane_objects() -> None:
    labels = np.zeros((3, 24, 24), dtype=np.int32)
    labels[0, 2:6, 2:6] = 1
    labels[0, 10:16, 10:16] = 2
    labels[1, 11:17, 11:17] = 2

    filtered, record = filter_labels_by_z_extent(labels, min_z_planes=2)

    assert record["removed_count"] == 1
    assert record["kept_count"] == 1
    assert filtered.max() == 1
    assert filtered[0, 3, 3] == 0
    assert filtered[0, 12, 12] == filtered[1, 13, 13] == 1


def test_auto3d_scoring_penalizes_extreme_merged_object() -> None:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[:, 5:55, 5:55] = 1
    label_id = 2
    for y in range(0, 64, 8):
        for x in range(0, 64, 8):
            if labels[0, y : y + 2, x : x + 2].any():
                continue
            labels[0, y : y + 2, x : x + 2] = label_id
            label_id += 1
    image[labels > 0] = 100.0

    metrics, warnings, score = rank_segmentation_labels(image, labels)

    assert metrics["largest_to_median_object_ratio"] > 1000
    assert score < 60
    assert any("extreme merged object" in w for w in warnings)


def test_score_roi_quality_zero_objects_is_floor() -> None:
    warnings: list[str] = []
    score = score_roi_quality(
        {"n_objects": 0}, warnings, noise_sigma=1.0, ndim=2, has_multiple_z=False
    )
    assert score == -1000.0
    assert any("zero" in w for w in warnings)


def test_score_roi_quality_penalizes_over_wide_mask() -> None:
    base = {"n_objects": 5, "top_bright_outside_fraction": 0.0}
    narrow = score_roi_quality(
        {**base, "mask_fraction": 0.05}, [], noise_sigma=1.0, ndim=2, has_multiple_z=False
    )
    wide = score_roi_quality(
        {**base, "mask_fraction": 0.6}, [], noise_sigma=1.0, ndim=2, has_multiple_z=False
    )
    assert wide < narrow  # too-wide mask (background included) is penalised


def test_score_roi_quality_penalizes_signal_outside_roi() -> None:
    base = {"n_objects": 5, "mask_fraction": 0.1}
    clean = score_roi_quality(
        {**base, "top_bright_outside_fraction": 0.0}, [], noise_sigma=1.0, ndim=2,
        has_multiple_z=False,
    )
    leaky = score_roi_quality(
        {**base, "top_bright_outside_fraction": 0.5}, [], noise_sigma=1.0, ndim=2,
        has_multiple_z=False,
    )
    assert leaky < clean  # bright signal outside the ROI (too narrow) is penalised


def test_confidence_from_score_tiers() -> None:
    good = {"n_objects": 12, "mask_fraction": 0.1, "largest_to_median_object_ratio": 2.0}
    assert confidence_from_score(90.0, good) == "high"
    assert confidence_from_score(65.0, good) == "medium"
    assert confidence_from_score(40.0, good) == "low"
    # zero objects and region-level merges are always low regardless of score (H1)
    assert confidence_from_score(95.0, {"n_objects": 0}) == "low"
    region = {"n_objects": 3, "mask_fraction": 0.2, "largest_to_median_object_ratio": 30.0}
    assert confidence_from_score(95.0, region) == "low"


def test_auto3d_confidence_accepts_stable_same_strategy_scores() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    candidates = [
        SegmentationCandidate("a", "direct_3d", labels, {}, {"n_objects": 1}, [], 80.0),
        SegmentationCandidate("b", "direct_3d", labels, {}, {"n_objects": 1}, [], 79.8),
    ]

    assert selection_confidence(candidates) == "high"


def test_auto3d_confidence_rejects_region_level_merged_labels() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    candidates = [
        SegmentationCandidate(
            "merged",
            "direct_3d",
            labels,
            {},
            {
                "n_objects": 12,
                "mask_fraction": 0.2,
                "largest_to_median_object_ratio": 20.0,
            },
            [],
            85.0,
        ),
        SegmentationCandidate(
            "merged_alt",
            "direct_3d",
            labels,
            {},
            {
                "n_objects": 10,
                "mask_fraction": 0.18,
                "largest_to_median_object_ratio": 18.0,
            },
            [],
            84.0,
        ),
    ]

    assert selection_confidence(candidates) == "low"


def test_segment_3d_cells_auto_plane_stitch_outputs_3d_labels(viewer) -> None:
    img = np.zeros((3, 64, 64), dtype=np.float32)
    img[0, 20:32, 20:32] = 80.0
    img[1, 21:33, 21:33] = 100.0
    img[2, 22:34, 22:34] = 90.0
    viewer.add_image(img, name="stack_target", scale=(0.8, 0.3, 0.3), metadata={"axes": "ZYX"})

    res = segment.segment_3d_cells_auto(
        "stack_target",
        candidate_modes=["plane_stitch"],
        background_radius=8,
        min_size=20,
        smoothing_sigma=0,
        fill_holes=False,
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert labels.shape == img.shape
    assert res["selected_strategy"] == "plane_stitch"
    assert res["n_objects"] == 1
    assert labels[0, 25, 25] == labels[1, 26, 26] == labels[2, 27, 27] != 0
    assert res["single_plane_object_fraction"] == pytest.approx(0.0)


def test_segment_3d_cells_auto_filters_single_plane_noise(viewer) -> None:
    img = np.zeros((3, 64, 64), dtype=np.float32)
    img[0, 20:32, 20:32] = 80.0
    img[1, 21:33, 21:33] = 100.0
    img[2, 22:34, 22:34] = 90.0
    img[0, 48:54, 48:54] = 120.0
    viewer.add_image(
        img,
        name="stack_with_noise",
        scale=(0.8, 0.3, 0.3),
        metadata={"axes": "ZYX"},
    )

    res = segment.segment_3d_cells_auto(
        "stack_with_noise",
        candidate_modes=["plane_stitch"],
        background_radius=8,
        min_size=10,
        smoothing_sigma=0,
        fill_holes=False,
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    best = res["candidate_summaries"][0]
    assert res["n_objects"] == 1
    assert labels[:, 50, 50].max() == 0
    assert best["params"]["z_extent_filter"]["removed_count"] >= 1


def test_segment_target_objects_keeps_cluster_without_split(viewer) -> None:
    image = np.zeros((96, 96), dtype=np.float32)
    image[25:55, 22:50] = 80.0
    image[35:62, 48:72] = 85.0
    viewer.add_image(image, name="touching")

    res = segment.segment_target_objects(
        "touching",
        background_radius=20,
        min_size=50,
        smoothing_sigma=0,
        split_touching=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["object_unit"] == "object_or_roi"
    assert res["n_objects"] == 1
    assert labels[40, 35] == labels[45, 60] != 0


def test_threshold_noise_floor_returns_value_above_dark_region() -> None:
    rng = np.random.default_rng(42)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :100] = rng.normal(10.0, 1.0, (200, 100))
    img[:, 100:] = 100.0

    t = segment._threshold_noise_floor(img, k_mad=5.0, dark_percentile=10.0)

    assert 10.0 < t < 25.0, f"threshold {t} should sit above dark median + a few sigma"
    assert t < 100.0, "threshold must stay below the bright region"


def test_threshold_noise_floor_handles_constant_image() -> None:
    img = np.full((50, 50), 7.0, dtype=np.float32)
    t = segment._threshold_noise_floor(img, k_mad=5.0, dark_percentile=10.0)
    assert t == pytest.approx(7.0)


def test_threshold_noise_floor_ignores_non_finite() -> None:
    img = np.full((50, 50), np.nan, dtype=np.float32)
    img[:25, :25] = 5.0
    t = segment._threshold_noise_floor(img, k_mad=3.0, dark_percentile=20.0)
    assert np.isfinite(t)
    assert t >= 5.0


def test_intersect_labels_with_mask_zeros_outside() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2

    mask = np.zeros_like(labels, dtype=bool)
    mask[0:5, 0:5] = True

    out = segment._intersect_labels_with_mask(labels, mask)

    assert (out == 1).sum() == (labels == 1).sum()
    assert (out == 2).sum() == 0


def test_intersect_labels_with_mask_renumbers_when_requested() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3] = 5
    labels[1:3, 4:6] = 9

    mask = np.ones_like(labels, dtype=bool)

    out = segment._intersect_labels_with_mask(labels, mask, renumber=True)

    unique = sorted(np.unique(out).tolist())
    assert unique == [0, 1, 2]


def test_segment_expression_domain_captures_dim_and_bright_regions(viewer) -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 30.0
    img[100:160, 100:160] += 8.0

    viewer.add_image(img, name="reporter")

    res = segment.segment_expression_domain(
        "reporter",
        k_mad=5.0,
        dark_percentile=10.0,
        min_area_um2=1.0,
    )

    assert res["empty_mask"] is False
    assert res["n_components"] >= 2
    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert labels[60, 60] > 0, "bright region must be inside domain"
    assert labels[130, 130] > 0, "dim region must also be inside domain"
    assert labels[10, 190] == 0, "background must be excluded"


def test_segment_expression_domain_empty_mask_for_pure_noise(viewer) -> None:
    rng = np.random.default_rng(1)
    img = rng.normal(5.0, 1.0, (100, 100)).astype(np.float32)
    viewer.add_image(img, name="noise_only")

    res = segment.segment_expression_domain(
        "noise_only",
        k_mad=20.0,
        dark_percentile=10.0,
    )

    assert res["empty_mask"] is True
    assert res["n_components"] == 0


def test_segment_expression_domain_labels_layer_naming(viewer) -> None:
    img = np.zeros((50, 50), dtype=np.float32)
    img[10:40, 10:40] = 100.0
    viewer.add_image(img, name="my_reporter")

    res = segment.segment_expression_domain("my_reporter", k_mad=3.0)

    assert res["labels_layer"] == "my_reporter_domain"
    assert "my_reporter_domain" in [L.name for L in viewer.layers]


def test_segment_expression_domain_intersects_with_nuclear_counterstain(viewer) -> None:
    rng = np.random.default_rng(2)
    reporter = np.zeros((200, 200), dtype=np.float32)
    reporter[:, :] = rng.normal(5.0, 1.0, reporter.shape)
    reporter[20:180, 20:180] += 30.0

    counterstain = np.zeros_like(reporter)
    counterstain[40:80, 40:80] = 200.0
    counterstain[100:140, 100:140] = 200.0

    viewer.add_image(reporter, name="reporter")
    viewer.add_image(counterstain, name="topro")

    res = segment.segment_expression_domain(
        "reporter",
        k_mad=5.0,
        counterstain_layer="topro",
        is_nuclear=True,
        counterstain_dilation_um=0.0,
        min_area_um2=0.0,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["counterstain_used"] is True
    assert res["counterstain_warnings"] == []
    assert labels[60, 60] > 0
    assert labels[120, 120] > 0
    assert labels[90, 90] == 0, "between nuclei must be excluded by counterstain"


def test_segment_expression_domain_3d_filters_noise_and_merges_domain(viewer) -> None:
    rng = np.random.default_rng(9)
    reporter = rng.normal(5.0, 0.7, (10, 64, 64)).astype(np.float32)
    reporter[3:7, 20:42, 22:44] += 35.0
    coords = (
        rng.integers(0, reporter.shape[0], size=300),
        rng.integers(0, reporter.shape[1], size=300),
        rng.integers(0, reporter.shape[2], size=300),
    )
    reporter[coords] += 80.0

    viewer.add_image(reporter, name="rep3d", scale=(1.0, 0.5, 0.5))

    res = segment.segment_expression_domain(
        "rep3d",
        k_mad=5.0,
        min_area_um2=8.0,
        smooth_sigma_um=0.5,
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert labels.shape == reporter.shape
    assert labels.max() == 1
    assert res["domain_label_count"] == 1
    assert res["n_components"] <= 8
    assert labels[4, 30, 30] > 0
    assert labels[0, 0, 0] == 0
    assert res["domain_volume_um3"] is not None


def test_segment_expression_domain_skips_non_nuclear_counterstain(viewer) -> None:
    reporter = np.zeros((100, 100), dtype=np.float32)
    reporter[20:80, 20:80] = 50.0
    counterstain = np.zeros_like(reporter)
    counterstain[10:90, 10:90] = 200.0

    viewer.add_image(reporter, name="rep2")
    viewer.add_image(counterstain, name="actin")

    res = segment.segment_expression_domain(
        "rep2",
        k_mad=3.0,
        counterstain_layer="actin",
        is_nuclear=False,
        min_area_um2=0.0,
    )

    assert res["counterstain_used"] is False
    assert any(
        "non-nuclear" in w for w in res["counterstain_warnings"]
    )


def test_segment_target_objects_boundary_mask_keeps_only_inside(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    img[60:80, 60:80] = 200.0
    viewer.add_image(img, name="img")

    boundary = np.zeros((100, 100), dtype=np.int32)
    boundary[0:50, 0:50] = 1
    viewer.add_labels(boundary, name="boundary")

    res = segment.segment_target_objects(
        "img",
        boundary_mask="boundary",
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert (labels[10:30, 10:30] > 0).any(), "object inside boundary kept"
    assert (labels[60:80, 60:80] == 0).all(), "object outside boundary dropped"


def test_segment_target_objects_boundary_mask_does_not_fill_domain(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[20:80, 20:80] = 20.0
    img[42:54, 42:54] = 120.0
    viewer.add_image(img, name="domain_relative")

    boundary = np.zeros((100, 100), dtype=np.int32)
    boundary[20:80, 20:80] = 1
    viewer.add_labels(boundary, name="domain_mask")

    res = segment.segment_target_objects(
        "domain_relative",
        boundary_mask="domain_mask",
        background_radius=0,
        min_size=20,
        smoothing_sigma=0,
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["threshold_scope"] == "boundary_mask"
    assert labels[48, 48] > 0, "bright region inside domain should be kept"
    assert labels[25, 25] == 0, "dim domain background must not become region"
    assert res["object_area_max"] < int(np.count_nonzero(boundary))


def test_segment_target_objects_default_unchanged(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 200.0
    viewer.add_image(img, name="reg")

    res = segment.segment_target_objects("reg", save_qc_png=False)

    assert res["n_objects"] >= 1
    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert (labels[40:60, 40:60] > 0).any()


def test_qc_png_renders_secondary_outline(tmp_path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:30, 20:30] = 100.0
    primary = np.zeros((64, 64), dtype=np.int32)
    primary[22:28, 22:28] = 1
    secondary = np.zeros((64, 64), dtype=np.int32)
    secondary[15:35, 15:35] = 1

    out_path = tmp_path / "two_tier_qc.png"
    segment._write_segmentation_qc_png(
        image,
        primary,
        out_path,
        secondary_outline_mask=secondary,
    )

    assert out_path.exists()
    rgb = np.asarray(Image.open(out_path))
    cyan_pixels = (
        (rgb[..., 1] > 150)
        & (rgb[..., 2] > 150)
        & (rgb[..., 0] < 100)
    )
    assert cyan_pixels.any(), "secondary outline should render in cyan"


def test_segment_target_objects_qc_includes_boundary_outline(viewer, tmp_path) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:50, 30:50] = 200.0
    viewer.add_image(img, name="rep_q")

    boundary = np.zeros((100, 100), dtype=np.int32)
    boundary[20:60, 20:60] = 1
    viewer.add_labels(boundary, name="b_q")

    out_path = tmp_path / "two_tier_qc.png"
    res = segment.segment_target_objects(
        "rep_q",
        boundary_mask="b_q",
        save_qc_png=True,
        qc_png_path=str(out_path),
    )

    assert res["qc_png_path"] is not None
    rgb = np.asarray(Image.open(out_path))
    cyan_pixels = (rgb[..., 1] > 150) & (rgb[..., 2] > 150) & (rgb[..., 0] < 100)
    assert cyan_pixels.any(), "domain outline must appear in cyan in Tier-2 QC PNG"


def test_segment_qc_writes_only_to_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools._segmentation_outputs import _default_qc_png_path

    reset_process_bundle()
    try:
        bundle = start_analysis(name="seg")
        path = _default_qc_png_path("foo_labels")
        assert path.parent == bundle / "qc"
        assert path.name == "foo_labels.png"
        # No flat segmentation_qc folder anywhere under the results root.
        assert not (tmp_path / "segmentation_qc").exists()
    finally:
        reset_process_bundle()
