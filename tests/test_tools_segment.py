from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imajin.project import create_project
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


def test_segmentation_qc_png_defaults_to_project_reports(viewer, tmp_path) -> None:
    create_project(tmp_path / "project")
    image = np.zeros((256, 256), dtype=np.float32)
    image[90:105, 100:115] = 100
    viewer.add_image(image, name="target")

    res = segment.segment_target_objects(
        "target",
        background_radius=16,
        min_size=20,
        smoothing_sigma=0,
    )

    out = tmp_path / "project" / "reports" / "segmentation_qc"
    assert res["qc_png_path"].startswith(str(out))
    assert (tmp_path / "project" / "reports" / "manifest.jsonl").exists()
    assert Image.open(res["qc_png_path"]).mode == "RGB"


def test_segmentation_qc_png_skips_tiny_default_outputs(viewer, tmp_path) -> None:
    create_project(tmp_path / "project")
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
    out = tmp_path / "project" / "reports" / "segmentation_qc"
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
