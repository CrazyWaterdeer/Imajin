"""Phase-2 spec coverage: target-channel resolution, physical units, and the
high-level analyze_target_cells workflow. Segmentation here is fed real labels
from a fixture — the Cellpose-SAM call is monkeypatched so these tests stay in
the fast suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imajin import session as state
from imajin.tools import channels, measure, workflows


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def _two_label_image() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    img = np.zeros_like(labels, dtype=np.float32)
    img[2:8, 2:8] = 100.0
    img[12:18, 12:18] = 50.0
    return labels, img


def _stub_cellpose(monkeypatch: pytest.MonkeyPatch, mask: np.ndarray) -> None:
    """Make cellpose_sam return a precomputed mask without touching torch/cellpose."""

    def _fake_eval(self, data, **kwargs):  # noqa: ANN001
        return mask, None, None

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return mask, None, None

    monkeypatch.setattr("imajin.tools._segmentation_io._get_cellpose_model", lambda *a, **kw: _FakeModel())


# --- Channel-to-analysis workflow -------------------------------------------------


def test_resolve_target_uses_confirmed_annotation(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="green_layer")
    viewer.add_image(np.zeros((4, 4), np.uint16), name="red_layer")
    state.put_channel_annotation("green_layer", role="target", color="green")

    result = state.resolve_target_channel()
    assert result.layer == "green_layer"
    assert result.source == "annotation"


def test_resolve_target_via_color_phrase(viewer) -> None:
    viewer.add_image(
        np.zeros((4, 4), np.uint16),
        name="ch_488",
        metadata={"channel_metadata": [{"color": "green"}], "channel_names": ["ch_488"]},
    )
    viewer.add_image(np.zeros((4, 4), np.uint16), name="ch_dapi")

    result = state.resolve_target_channel("green")
    assert result.layer == "ch_488"
    assert result.source in {"phrase", "annotation"}


def test_resolve_target_ambiguous_raises(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="img_a")
    viewer.add_image(np.zeros((4, 4), np.uint16), name="img_b")

    with pytest.raises(state.AmbiguousChannelError):
        state.resolve_target_channel()


def test_resolve_target_skips_counterstain(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="dapi")
    viewer.add_image(np.zeros((4, 4), np.uint16), name="gfp")
    state.put_channel_annotation("dapi", role="counterstain", color="uv")

    # Only one image layer (gfp) is selectable since dapi is counterstain.
    result = state.resolve_target_channel()
    assert result.layer == "gfp"
    assert result.source == "inference"


def test_resolve_target_refuses_explicit_counterstain(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="dapi")
    state.put_channel_annotation("dapi", role="counterstain", color="uv")

    with pytest.raises(state.AmbiguousChannelError):
        state.resolve_target_channel("uv")


def test_resolve_target_channel_tool_returns_dict(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="solo")
    out = channels.resolve_target_channel_tool()
    assert out["ok"] is True
    assert out["layer"] == "solo"
    assert out["source"] == "inference"


def test_resolve_target_channel_tool_returns_error_payload(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="img_a")
    viewer.add_image(np.zeros((4, 4), np.uint16), name="img_b")

    out = channels.resolve_target_channel_tool()
    assert out["ok"] is False
    assert "candidates" in out
    assert set(out["candidates"]) == {"img_a", "img_b"}


# --- Physical-unit measurement columns -------------------------------------------


def test_measure_intensity_adds_area_px_without_scale(viewer) -> None:
    labels, img = _two_label_image()
    viewer.add_labels(labels, name="masks")
    viewer.add_image(img, name="ch")

    res = measure.measure_intensity("masks", ["ch"])
    df = state.get_table(res["table_name"])

    assert "area_px" in df.columns
    assert "area_um2" not in df.columns
    assert res["voxel_scale"] is None
    assert res["has_physical_units"] is False


def test_measure_intensity_2d_adds_area_um2(viewer) -> None:
    labels, img = _two_label_image()
    viewer.add_labels(labels, name="masks", scale=(0.5, 0.5))
    viewer.add_image(img, name="ch", scale=(0.5, 0.5))

    res = measure.measure_intensity("masks", ["ch"])
    df = state.get_table(res["table_name"])

    assert "area_px" in df.columns
    assert "area_um2" in df.columns
    assert res["has_physical_units"] is True
    assert (df["area_um2"] / df["area_px"]).round(6).eq(0.25).all()
    assert "centroid_y_um" in df.columns
    assert "centroid_x_um" in df.columns


def test_measure_intensity_3d_adds_volume_columns(viewer) -> None:
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    labels[1:3, 2:5, 2:5] = 1
    img = labels.astype(np.float32) * 100.0
    viewer.add_labels(labels, name="vol_masks", scale=(0.5, 0.2, 0.2))
    viewer.add_image(img, name="vol_ch", scale=(0.5, 0.2, 0.2))

    res = measure.measure_intensity("vol_masks", ["vol_ch"])
    df = state.get_table(res["table_name"])

    assert "volume_voxels" in df.columns
    assert "volume_um3" in df.columns
    assert "centroid_z_um" in df.columns
    voxel_volume = 0.5 * 0.2 * 0.2
    assert df["volume_um3"].iloc[0] == pytest.approx(
        df["volume_voxels"].iloc[0] * voxel_volume
    )


# --- Time-course columns ---------------------------------------------------------


def test_time_course_adds_time_index_and_time_s_when_interval_present(viewer) -> None:
    labels, img = _two_label_image()
    series = np.stack([img, img * 2, img * 3], axis=0)
    viewer.add_labels(labels, name="rois")
    viewer.add_image(
        series,
        name="movie",
        metadata={"axes": "TYX", "time_interval_s": 1.5},
    )

    res = measure.measure_intensity_over_time("rois", "movie")
    df = state.get_table(res["table_name"])

    assert res["time_interval_s"] == 1.5
    assert "time_index" in df.columns
    assert "time_s" in df.columns
    expected_seconds = sorted({0.0, 1.5, 3.0})
    assert sorted(df["time_s"].unique().tolist()) == expected_seconds


def test_time_course_omits_time_s_without_interval(viewer) -> None:
    labels, img = _two_label_image()
    series = np.stack([img, img * 2], axis=0)
    viewer.add_labels(labels, name="rois")
    viewer.add_image(series, name="movie", metadata={"axes": "TYX"})

    res = measure.measure_intensity_over_time("rois", "movie")
    df = state.get_table(res["table_name"])
    assert "time_index" in df.columns
    assert "time_s" not in df.columns
    assert res["time_interval_s"] is None


# --- analyze_target_cells workflow -----------------------------------------------


def test_analyze_target_cells_full_path(viewer, monkeypatch) -> None:
    labels, img = _two_label_image()
    viewer.add_image(img, name="green_target", scale=(0.5, 0.5))
    state.put_channel_annotation("green_target", role="target", color="green")

    _stub_cellpose(monkeypatch, labels)

    res = workflows.analyze_target_cells()
    assert res["ok"] is True
    assert res["target_channel"] == "green_target"
    assert res["target_source"] == "annotation"
    assert res["segmentation_method"] == "target_objects"
    assert res["n_objects"] == 2
    assert res["has_physical_units"] is True
    assert res["voxel_scale"] == [0.5, 0.5]
    df = state.get_table(res["table_name"])
    assert "mean_intensity_green_target" in df.columns
    assert "area_um2" in df.columns


def test_analyze_target_cells_keeps_z_stack_measurement_3d(viewer) -> None:
    img = np.zeros((4, 64, 64), dtype=np.float32)
    img[1:4, 20:32, 22:34] = 100.0
    viewer.add_image(
        img,
        name="vol_target",
        scale=(1.0, 0.4, 0.4),
        metadata={"axes": "ZYX"},
    )

    res = workflows.analyze_target_cells(
        target="vol_target",
        segmentation_options={
            "background_radius": 8,
            "min_size": 20,
            "smoothing_sigma": 0,
        },
    )

    assert res["ok"] is True
    assert res["analysis_dim"] == "3d"
    assert res["do_3D"] is True
    assert "volume_um3" in res["table_columns"]
    df = state.get_table(res["table_name"])
    assert df["volume_um3"].iloc[0] > 0


def test_analyze_target_cells_accepts_auto_3d_cells_alias(viewer) -> None:
    img = np.zeros((3, 64, 64), dtype=np.float32)
    img[0, 20:32, 20:32] = 80.0
    img[1, 21:33, 21:33] = 100.0
    img[2, 22:34, 22:34] = 90.0
    viewer.add_image(
        img,
        name="auto3d_target",
        scale=(0.8, 0.3, 0.3),
        metadata={"axes": "ZYX"},
    )

    res = workflows.analyze_target_cells(
        target="auto3d_target",
        segmentation_method="auto_3d_cells",
        segmentation_options={
            "candidate_modes": ["plane_stitch"],
            "background_radius": 8,
            "min_size": 20,
            "smoothing_sigma": 0,
            "fill_holes": False,
            "save_qc_png": False,
        },
    )

    assert res["ok"] is True
    assert res["segmentation_method"] == "auto_3d_cells"
    assert res["analysis_dim"] == "3d"
    assert res["n_objects"] == 1
    assert "volume_um3" in res["table_columns"]


def test_analyze_target_cells_accepts_segment_intensity_regions_alias(viewer) -> None:
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:8, 2:8] = 100.0
    img[12:18, 12:18] = 50.0
    viewer.add_image(img, name="calexa")

    res = workflows.analyze_target_cells(
        target="calexa",
        segmentation_method="segment_intensity_regions",
        segmentation_options={"min_size": 4, "smoothing_sigma": 0},
    )

    assert res["ok"] is True
    assert res["segmentation_method"] == "intensity_regions"
    assert res["n_objects"] == 2


def test_analyze_target_cells_saves_result_bundle(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    img = np.zeros((256, 256), dtype=np.float32)
    img[80:95, 90:105] = 100.0
    img[150:168, 140:158] = 80.0
    viewer.add_image(img, name="green_target", scale=(0.5, 0.5))

    res = workflows.analyze_target_cells(target="green_target")

    assert res["ok"] is True
    assert res["result_bundle_path"].startswith(
        str(tmp_path / "results")
    )
    assert res["result_files"]["labels_cells"] == "labels/cells/green_target.tif"
    assert res["result_files"]["labels_domain"] is None
    assert res["result_files"]["qc_png"] is not None
    bundle = Path(res["result_bundle_path"])
    assert (bundle / "labels" / "cells" / "green_target.tif").exists()
    assert (bundle / "tables" / "combined.csv").exists()


def test_analyze_target_cells_bundle_lands_in_layer_anchor_folder(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))

    anchor = tmp_path / "raw_data"
    anchor.mkdir()
    source = anchor / "reporter.lsm"
    source.write_bytes(b"")

    img = np.random.default_rng(0).normal(5.0, 0.2, (300, 300)).astype(np.float32)
    img[80:220, 90:230] = 200.0
    viewer.add_image(
        img,
        name="reporter",
        scale=(0.5, 0.5),
        metadata={"path": str(source)},
    )
    state.put_channel_annotation("reporter", role="target", color="green")

    res = workflows.analyze_target_cells(target="reporter")

    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    assert bundle.parent == anchor.resolve()
    assert bundle.name.endswith("__single")
    assert str(res["qc_png_path"]).startswith(str(anchor.resolve()))
    assert Path(res["qc_png_path"]).exists()
    assert not (anchor / "segmentation_qc").exists()
    # Note: after T10, the segmentation step writes the QC PNG into an adhoc
    # process bundle under IMAJIN_RESULTS_DIR before the anchor bundle is created.
    # The fallback dir may therefore exist; the important thing is that the
    # *final* QC PNG and the result bundle land in the anchor folder (above).
    assert (bundle / "labels" / "cells" / "reporter.tif").exists()

    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["run_context"]["folder_set"] == [str(anchor.resolve())]
    assert meta["run_context"]["channel_roles"] == {"reporter": "target"}
    assert meta["run_context"]["scope_filters"] == []


def test_analyze_target_cells_with_explicit_target(viewer, monkeypatch) -> None:
    labels, img = _two_label_image()
    viewer.add_image(img, name="ch1")
    viewer.add_image(np.zeros_like(img), name="ch2")
    _stub_cellpose(monkeypatch, labels)

    res = workflows.analyze_target_cells(target="ch1")
    assert res["ok"] is True
    assert res["target_channel"] == "ch1"
    assert res["target_source"] == "explicit"


def test_analyze_target_cells_reports_zero_objects(viewer, monkeypatch) -> None:
    _, img = _two_label_image()
    viewer.add_image(img, name="solo")
    empty_mask = np.zeros_like(img, dtype=np.int32)
    _stub_cellpose(monkeypatch, empty_mask)

    res = workflows.analyze_target_cells(segmentation_method="cellpose_sam")
    assert res["ok"] is False
    assert res["stage"] == "segment"
    assert "zero objects" in res["error"]


def test_analyze_target_cells_returns_error_when_target_ambiguous(viewer) -> None:
    viewer.add_image(np.zeros((4, 4), np.uint16), name="a")
    viewer.add_image(np.zeros((4, 4), np.uint16), name="b")

    res = workflows.analyze_target_cells()
    assert res["ok"] is False
    assert res["stage"] == "resolve_target"
    assert set(res["candidates"]) == {"a", "b"}


def test_analyze_target_cells_warns_on_no_voxel_size(viewer, monkeypatch) -> None:
    labels, img = _two_label_image()
    viewer.add_image(img, name="solo")
    _stub_cellpose(monkeypatch, labels)

    res = workflows.analyze_target_cells()
    assert res["ok"] is True
    assert any("voxel size" in w for w in res["warnings"])


def test_analyze_target_cells_skips_counterstain_unless_explicit(
    viewer, monkeypatch
) -> None:
    labels, img = _two_label_image()
    viewer.add_image(img, name="dapi")
    viewer.add_image(img * 2, name="gfp")
    state.put_channel_annotation("dapi", role="counterstain", color="uv")

    _stub_cellpose(monkeypatch, labels)

    res = workflows.analyze_target_cells()
    assert res["ok"] is True
    assert res["target_channel"] == "gfp"


def test_derive_size_params_with_diameter() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=15.0, voxel_spacing=(0.5, 0.5))

    assert out["min_distance_um"] == pytest.approx(15.0 * 0.7)
    assert out["min_area_um2"] == pytest.approx(np.pi * (15.0 / 4) ** 2)
    assert out["cellpose_diameter_px"] == pytest.approx(15.0 / 0.5)


def test_derive_size_params_returns_empty_when_diameter_none() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=None, voxel_spacing=(0.5, 0.5))
    assert out == {}


def test_derive_size_params_handles_missing_voxel() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=10.0, voxel_spacing=None)
    assert out["min_distance_um"] == pytest.approx(7.0)
    assert "cellpose_diameter_px" not in out


def test_analyze_target_cells_two_tier_produces_long_format(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    img[120:160, 120:160] += 12.0

    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "dark_percentile": 10.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )

    assert res["ok"] is True
    assert res["domain_layer"].endswith("_domain")
    assert res["n_domain_components"] >= 2
    assert "tier_table_name" in res

    from imajin.session import get_table
    table = get_table(res["tier_table_name"])
    assert "tier" in table.columns
    assert set(table["tier"].unique()) == {"domain", "cells"}
    domain_rows = table[table["tier"] == "domain"]
    cell_rows = table[table["tier"] == "cells"]
    assert len(domain_rows) == 1
    assert len(cell_rows) >= 1


def test_analyze_target_cells_single_tier_unchanged(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 200.0
    viewer.add_image(img, name="single_tier", scale=(0.5, 0.5))

    res = analyze_target_cells(target="single_tier")

    assert res["ok"] is True
    assert "domain_layer" not in res or res.get("domain_layer") is None
    assert "tier_table_name" not in res or res.get("tier_table_name") is None


def test_two_tier_keeps_active_region_inside_larger_domain(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(7)
    h, w = 256, 256
    img = rng.normal(5.0, 1.0, (h, w)).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    cluster_mask = ((yy - 60) ** 2 + (xx - 60) ** 2) < 40 ** 2
    halo_intensity = np.maximum(
        0,
        40.0 - 0.6 * np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2),
    )
    img += halo_intensity
    img[cluster_mask] = 250.0  # saturated core

    viewer.add_image(img, name="reporter_long", scale=(0.5, 0.5))

    single = analyze_target_cells(target="reporter_long")
    assert single["ok"] is True
    single_labels = np.asarray(viewer.layers[single["labels_layer"]].data)
    single_area = int((single_labels > 0).sum())

    # Add the same image again so the workflow operates on an independent copy
    viewer.add_image(img, name="reporter_long_two", scale=(0.5, 0.5))
    two_tier = analyze_target_cells(
        target="reporter_long_two",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert two_tier["ok"] is True
    assert two_tier["segmentation_threshold_scope"] == "boundary_mask"
    cell_labels = np.asarray(viewer.layers[two_tier["cells_layer"]].data)
    cell_area = int((cell_labels > 0).sum())

    domain_labels = np.asarray(viewer.layers[two_tier["domain_layer"]].data)
    domain_area = int((domain_labels > 0).sum())
    assert domain_area > single_area, "domain should capture the soft halo"
    assert 0 < cell_area < domain_area, (
        "active region must be a thresholded subset of the expression domain"
    )


def test_two_tier_region_mask_constrains_both_tiers(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    img = np.zeros((200, 200), dtype=np.float32)
    img[30:60, 30:60] = 250.0  # inside the ROI
    img[140:170, 140:170] = 250.0  # outside the ROI
    viewer.add_image(img, name="rep_region", scale=(0.5, 0.5))
    roi = np.zeros((200, 200), dtype=np.int32)
    roi[0:100, 0:100] = 1
    viewer.add_labels(roi, name="region")

    res = analyze_target_cells(
        target="rep_region",
        region_mask="region",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
    )
    assert res["ok"] is True
    cells = np.asarray(viewer.layers[res["cells_layer"]].data)
    domain = np.asarray(viewer.layers[res["domain_layer"]].data)
    assert (cells[140:170, 140:170] == 0).all(), "no cells outside the ROI"
    assert (domain[140:170, 140:170] == 0).all(), "no domain outside the ROI"
    assert (cells[30:60, 30:60] > 0).any(), "cells found inside the ROI"


def test_region_mask_rejected_for_method_without_boundary(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    viewer.add_image(np.zeros((64, 64), dtype=np.float32), name="x_rm")
    viewer.add_labels(np.ones((64, 64), dtype=np.int32), name="roi_x")
    res = analyze_target_cells(
        target="x_rm", region_mask="roi_x", segmentation_method="cellpose_sam"
    )
    assert res["ok"] is False and res["stage"] == "region_mask"


def test_region_mask_conflicts_with_explicit_boundary_mask(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    viewer.add_image(np.zeros((64, 64), dtype=np.float32), name="y_rm")
    viewer.add_labels(np.ones((64, 64), dtype=np.int32), name="roi_y")
    res = analyze_target_cells(
        target="y_rm",
        region_mask="roi_y",
        segmentation_options={"boundary_mask": "roi_y"},
    )
    assert res["ok"] is False and res["stage"] == "region_mask"


def test_analyze_target_cells_rerun_guard(viewer, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    img = np.zeros((80, 80), dtype=np.float32)
    img[20:50, 20:50] = 250.0
    viewer.add_image(img, name="rg_rep", scale=(0.5, 0.5))
    viewer.layers["rg_rep"].metadata["source_path"] = str(tmp_path / "rg_rep.lsm")

    first = workflows.analyze_target_cells(target="rg_rep")
    assert first["ok"] is True and not first.get("already_analysed")
    assert any(
        r["recipe_id"].startswith("interactive:") and r["status"] == "complete"
        for r in state.list_runs()
    ), "interactive analysis recorded a complete AnalysisRun"

    # Second call: the guard must short-circuit BEFORE any heavy step runs.
    def _boom(*a, **k):
        raise AssertionError("re-ran a finished analysis")

    monkeypatch.setattr(workflows, "_run_preprocess_step", _boom)
    monkeypatch.setattr(workflows, "_run_segmentation_step", _boom)
    monkeypatch.setattr(workflows, "_precompute_domain_layer", _boom)

    again = workflows.analyze_target_cells(target="rg_rep")
    assert again["ok"] is True and again["already_analysed"] is True
    assert "rerun=True" in again["message"]
    # rerun=True bypasses the guard -> _boom fires, proving it tried to recompute.
    with pytest.raises(AssertionError):
        workflows.analyze_target_cells(target="rg_rep", rerun=True)


def test_failed_run_does_not_block_retry(viewer, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:45, 20:45] = 300.0
    viewer.add_image(img, name="blank_rep", scale=(0.5, 0.5))
    viewer.layers["blank_rep"].metadata["source_path"] = str(tmp_path / "blank.lsm")

    # Force an empty mask: a min_size larger than the whole image removes every
    # object -> a recorded *failed* run.
    r1 = workflows.analyze_target_cells(
        target="blank_rep", segmentation_options={"min_size": 10_000_000}
    )
    assert r1["ok"] is False
    assert any(
        r["status"] == "failed" and r["recipe_id"].startswith("interactive:")
        for r in state.list_runs()
    )

    # Retry with normal params and no rerun: a failed run must not block.
    r2 = workflows.analyze_target_cells(target="blank_rep")
    assert not r2.get("already_analysed"), "a failed run must not block a retry"
    assert r2["ok"] is True


def test_analyze_target_cells_single_tier_writes_new_layout_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(target="reporter")
    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    assert bundle.exists()
    assert bundle.name.endswith("__single")
    assert (bundle / "labels" / "cells" / "reporter.tif").exists()
    assert not (bundle / "labels" / "domain").exists()
    if res["result_files"]["qc_png"] is None:
        assert not (bundle / "qc").exists()
    else:
        assert (bundle / res["result_files"]["qc_png"]).exists()

    meta = json.loads((bundle / "metadata.json").read_text())
    run_context = meta["run_context"]
    assert meta["schema_version"] == 3
    assert run_context["kind"] == "single"
    assert run_context["tier"] == "single_tier"
    assert run_context["status"] == "complete"
    assert len(run_context["samples"]) == 1
    assert run_context["samples"][0]["status"] == "complete"
    assert "outputs" not in run_context["samples"][0]


def test_analyze_target_cells_two_tier_writes_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    img[120:160, 120:160] += 12.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    assert bundle.exists()
    assert bundle.name.endswith("__two_tier")
    assert (bundle / "labels" / "cells" / "reporter.tif").exists()
    assert (bundle / "labels" / "domain" / "reporter.tif").exists()
    assert not (tmp_path / "segmentation_qc").exists()

    meta = json.loads((bundle / "metadata.json").read_text())
    run_context = meta["run_context"]
    assert meta["schema_version"] == 3
    assert run_context["kind"] == "single"
    assert run_context["tier"] == "two_tier"
    assert run_context["status"] == "complete"
    assert len(run_context["samples"]) == 1
    assert "outputs" not in run_context["samples"][0]


def test_analyze_target_cells_writes_into_active_parent_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    """When a parent bundle is active, no own-bundle is created."""
    from imajin.results import create_result_bundle
    from imajin.tools.results import with_active_bundle
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    parent = create_result_bundle(name="parent", kind="batch", tier="single_tier")

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    with with_active_bundle(parent):
        res = analyze_target_cells(target="reporter")

    assert res["ok"] is True
    assert res["result_bundle_path"] is None
    assert (parent / "labels" / "cells" / "reporter.tif").exists()
    # Registering the child's outputs normalises the parent's v1 seed to
    # schema_v3, so status now lives under run_context.
    meta = json.loads((parent / "metadata.json").read_text())
    run_context = meta.get("run_context") or meta
    assert run_context["status"] == "in_progress"
    # The batch runner owns the parent's sample records; the child writes none.
    assert not run_context.get("samples")


def test_analyze_target_cells_returns_primary_table_name_single_tier(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))
    res = analyze_target_cells(target="reporter")
    assert res["primary_table_name"] == res["table_name"]


def test_analyze_target_cells_returns_primary_table_name_two_tier(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))
    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert res["primary_table_name"] == res["tier_table_name"]


def _blob_image() -> np.ndarray:
    img = np.zeros((256, 256), dtype=np.float32)
    img[80:95, 90:105] = 100.0
    img[150:168, 140:158] = 80.0
    return img


def test_per_file_identity_comes_from_the_source_file_not_the_layer(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """Two files sharing a channel name must not share a bundle identity.

    Channel names come from instrument metadata (Ch2-T1) and repeat for every
    file in a folder. Deriving identity from the layer made all of a session's
    files collide on one slug, which silently overwrote label TIFFs once they
    shared a bundle.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    source = raw / "rectum_1.lsm"
    source.write_bytes(b"stub")

    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)

    res = workflows.analyze_target_cells(target="Ch2-T1")

    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    # Folder and per-sample outputs are named for the FILE, not the channel.
    assert bundle.name.endswith("_rectum_1__single")
    assert res["result_files"]["labels_cells"] == "labels/cells/rectum_1.tif"
    assert (bundle / "labels" / "cells" / "rectum_1.tif").exists()

    meta = json.loads((bundle / "metadata.json").read_text())
    sample = meta["run_context"]["samples"][0]
    assert sample["sample_name"] == "rectum_1"
    assert sample["source_file"] == str(source)
    assert sample["source_layer"] == "Ch2-T1"


def test_layer_without_source_file_still_uses_the_layer_name(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """In-memory layers have no file behind them; identity falls back cleanly."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    viewer.add_image(_blob_image(), name="synthetic", scale=(0.5, 0.5))

    res = workflows.analyze_target_cells(target="synthetic")

    assert res["ok"] is True
    assert res["result_files"]["labels_cells"] == "labels/cells/synthetic.tif"
    meta = json.loads((Path(res["result_bundle_path"]) / "metadata.json").read_text())
    assert meta["run_context"]["samples"][0]["sample_name"] == "synthetic"


def test_sequential_per_file_analyses_collect_in_one_session_bundle(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """The reported bug, end to end.

    A hand-drawn-ROI session steps through files one at a time: start_analysis
    once, then analyze_target_cells per file. Every file's outputs must land in
    that ONE folder. Before this, analyze_target_cells minted a folder per file
    and evicted the session bundle from the process slot, so seven files
    produced seven folders and orphaned the bundle the agent had opened.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    results_root = tmp_path / "results"
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(results_root))

    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools

    reset_process_bundle()
    opened = bundle_tools.start_analysis("hindgut_rectum_roi")
    session_bundle = Path(opened["bundle_path"])

    stems = ["rectum_1", "rectum_2", "rectum_3"]
    for stem in stems:
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        # Every file loads under the SAME channel name, as real instruments emit.
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)

        res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)
        assert res["ok"] is True
        assert Path(res["result_bundle_path"]) == session_bundle, (
            f"{stem} opened its own bundle instead of appending"
        )

    # Exactly one bundle folder exists — no per-file folders, no orphan.
    bundles = sorted(p.name for p in results_root.iterdir() if p.is_dir())
    assert bundles == [session_bundle.name]

    # Each file kept its own label TIFF; none overwrote another.
    written = sorted(p.stem for p in (session_bundle / "labels" / "cells").iterdir())
    assert written == stems

    meta = json.loads((session_bundle / "metadata.json").read_text())
    run_context = meta["run_context"]
    # Still open for more files — never finalized mid-session.
    assert run_context["status"] == "in_progress"
    assert [s["sample_name"] for s in run_context["samples"]] == stems
    assert [Path(s["source_file"]).name for s in run_context["samples"]] == [
        f"{stem}.lsm" for stem in stems
    ]
    reset_process_bundle()


def test_analysis_does_not_adopt_a_stray_adhoc_bundle(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """An ad-hoc bundle minted by a stray output must not swallow the analysis.

    This is the property commit 53b26a0 was protecting when it switched to the
    contextvar; the provenance check has to preserve it.
    """
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import (
        bundle_output_path,
        ensure_active_bundle,
        register_output,
        reset_process_bundle,
    )

    reset_process_bundle()
    adhoc = ensure_active_bundle()  # what a stray QC write would create
    stray = bundle_output_path("qc", "stray.png")
    stray.write_bytes(b"\x89PNG\r\n")
    register_output("qc_png", stray, {})

    viewer.add_image(_blob_image(), name="green_target", scale=(0.5, 0.5))
    res = workflows.analyze_target_cells(target="green_target")

    assert res["ok"] is True
    assert Path(res["result_bundle_path"]) != adhoc
    reset_process_bundle()


def test_analysis_does_not_append_to_a_finalized_bundle(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """A closed folder must not keep accepting new files' outputs."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools

    reset_process_bundle()
    opened = bundle_tools.start_analysis("closed_session")
    closed = Path(opened["bundle_path"])
    bundle_tools.finalize_analysis()

    viewer.add_image(_blob_image(), name="green_target", scale=(0.5, 0.5))
    res = workflows.analyze_target_cells(target="green_target")

    assert res["ok"] is True
    assert Path(res["result_bundle_path"]) != closed
    reset_process_bundle()


def test_qc_png_never_lands_in_a_previous_files_bundle(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """One file's outputs must not straddle two bundles.

    The QC PNG used to resolve through ensure_active_bundle() during
    segmentation — minutes before the analysis chose its bundle — so it landed
    in the PREVIOUS file's already-finalized folder. In the reported session
    every file's only informatively-named QC image was filed under the wrong
    sample.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    results_root = tmp_path / "results"
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(results_root))

    from imajin.result_bundles import reset_process_bundle

    reset_process_bundle()
    seen: list[Path] = []
    for stem in ["rectum_1", "rectum_2"]:
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)

        res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)
        assert res["ok"] is True
        seen.append(Path(res["result_bundle_path"]))

    assert seen[0] != seen[1]  # standalone runs still get their own folders
    for bundle, stem in zip(seen, ["rectum_1", "rectum_2"], strict=True):
        pngs = sorted(p.name for p in (bundle / "qc").iterdir())
        # Every QC image in this folder belongs to THIS file.
        assert pngs, f"{stem} bundle has no QC image"
        for name in pngs:
            assert stem in name, f"{name} in {bundle.name} belongs to another file"
    reset_process_bundle()


def test_session_bundle_accumulates_tables_and_a_pooled_combined_csv(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """A session bundle must be a complete result set on its own.

    The per-file measurements previously reached disk only if the agent
    separately called save_result_bundle with the table name, so the pooled
    table a session was actually analysing could be computed, plotted and then
    lost with the process — the one artifact that cannot be regenerated without
    redoing every ROI.
    """
    import pandas as pd

    raw = tmp_path / "raw"
    raw.mkdir()
    results_root = tmp_path / "results"
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(results_root))

    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools

    reset_process_bundle()
    session_bundle = Path(bundle_tools.start_analysis("pooled")["bundle_path"])

    stems = ["rectum_1", "rectum_2"]
    for stem in stems:
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)
        assert workflows.analyze_target_cells(target="Ch2-T1", rerun=True)["ok"]

    tables = sorted(p.name for p in (session_bundle / "tables").iterdir())
    assert tables == ["combined.csv", "rectum_1.csv", "rectum_2.csv"]

    combined = pd.read_csv(session_bundle / "tables" / "combined.csv")
    assert sorted(combined["sample_name"].unique()) == stems
    per_file = [
        pd.read_csv(session_bundle / "tables" / f"{stem}.csv") for stem in stems
    ]
    assert len(combined) == sum(len(df) for df in per_file)

    # The durable index knows both files, so the session can be resumed.
    from imajin.result_bundles import read_sample_index

    index = read_sample_index(session_bundle)
    assert sorted(e["key"] for e in index["entries"]) == [f"{s}.lsm" for s in stems]
    assert not index.get("legacy_inferred")
    reset_process_bundle()


def test_labels_are_written_once_per_sample(viewer, tmp_path, monkeypatch) -> None:
    """One writer per artifact.

    populate_sample_outputs wrote labels/cells/<sample>.tif and registered
    nothing, then save_result_bundle wrote the same volume again under the
    layer-derived name — every bundle carried two full-resolution copies of the
    identical array.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import results as results_tools

    reset_process_bundle()
    source = raw / "rectum_1.lsm"
    source.write_bytes(b"stub")
    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)

    res = workflows.analyze_target_cells(target="Ch2-T1")
    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])

    # The agent then saves the same labels layer, as the prompt tells it to.
    results_tools.save_result_bundle(
        "rectum_1", labels_layers=[res["labels_layer"]]
    )

    tifs = sorted(p.name for p in (bundle / "labels" / "cells").iterdir())
    assert tifs == ["rectum_1.tif"]

    meta = json.loads((bundle / "metadata.json").read_text())
    label_entries = [o for o in meta["outputs"] if o["kind"] == "labels_tiff"]
    assert len(label_entries) == 1
    assert label_entries[0]["path"] == "labels/cells/rectum_1.tif"
    reset_process_bundle()


def test_reopened_bundle_is_really_the_append_target(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """open_result_bundle promises the bundle is the append target — hold it to it.

    A finalized bundle is deliberately not reusable, so promoting one without
    re-opening it left every resumed file minting its own folder while the tool
    reported that outputs would land in the bundle.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle_resume

    reset_process_bundle()
    first = raw / "rectum_1.lsm"
    first.write_bytes(b"stub")
    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(first)
    res = workflows.analyze_target_cells(target="Ch2-T1")
    bundle = Path(res["result_bundle_path"])
    assert res["bundle_created"] is True

    # New session: reopen that bundle and analyse the next file.
    reset_process_bundle()
    bundle_resume.open_result_bundle(str(bundle))

    second = raw / "rectum_2.lsm"
    second.write_bytes(b"stub")
    viewer.layers.clear()
    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(second)
    res2 = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)

    assert Path(res2["result_bundle_path"]) == bundle
    assert res2["bundle_created"] is False
    tifs = sorted(p.stem for p in (bundle / "labels" / "cells").iterdir())
    assert tifs == ["rectum_1", "rectum_2"]
    reset_process_bundle()


def test_saving_after_each_file_never_clobbers_an_earlier_files_labels(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """analyze + save_result_bundle per file, the flow the prompt describes.

    save_result_bundle looked up "where is this labels layer already stored?" by
    LAYER NAME — the identity this whole series declares unreliable, because
    every file segments to `Ch2-T1_objects`. The lookup returned the FIRST
    file's TIFF, so each subsequent save overwrote it while metadata.json still
    attributed the file to the earlier sample.
    """
    import tifffile

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools
    from imajin.tools import results as results_tools

    reset_process_bundle()
    bundle = Path(bundle_tools.start_analysis("session")["bundle_path"])

    def image_with(n_blobs: int) -> np.ndarray:
        img = np.zeros((256, 256), dtype=np.float32)
        for y, x in [(80, 90), (150, 140), (40, 200), (200, 40)][:n_blobs]:
            img[y : y + 15, x : x + 15] = 100.0
        return img

    expected: dict[str, int] = {}
    for stem, n_blobs in (("rectum_1", 2), ("rectum_2", 4)):
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        viewer.add_image(image_with(n_blobs), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)

        res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)
        assert res["ok"] is True
        expected[stem] = int(res["n_objects"])
        saved = results_tools.save_result_bundle(
            stem, labels_layers=[res["labels_layer"]]
        )
        assert [Path(p).name for p in saved["outputs"]["labels"]] == [f"{stem}.tif"]

    assert expected["rectum_1"] != expected["rectum_2"], "test needs distinct data"
    for stem, n in expected.items():
        tif = bundle / "labels" / "cells" / f"{stem}.tif"
        assert int(tifffile.imread(tif).max()) == n, (
            f"{stem}.tif holds another file's labels"
        )


def test_combined_csv_does_not_double_count_a_saved_table(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """save_result_bundle(table_names=...) must not inflate the pooled table.

    combined.csv is rebuilt from the registered table_csv outputs. Counting the
    agent's own saved copies folded the same measurements in twice — and those
    rows carry no sample_name, so a group-by silently mis-buckets them.
    """
    import pandas as pd

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools
    from imajin.tools import results as results_tools

    reset_process_bundle()
    bundle = Path(bundle_tools.start_analysis("session")["bundle_path"])

    stems = ["rectum_1", "rectum_2"]
    for stem in stems:
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)
        res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)
        # The agent also persists the table, exactly as the prompt instructs.
        results_tools.save_result_bundle(stem, table_names=[res["table_name"]])

    combined = pd.read_csv(bundle / "tables" / "combined.csv")
    per_file = [pd.read_csv(bundle / "tables" / f"{s}.csv") for s in stems]
    assert len(combined) == sum(len(df) for df in per_file)
    assert combined["sample_name"].notna().all()
    assert sorted(combined["sample_name"].unique()) == stems
    reset_process_bundle()


def test_a_failed_analysis_does_not_capture_the_next_file(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """A per-file folder is not a session, even while it is still open.

    Pinning the destination before segmentation means a file whose analysis
    fails still mints a folder and leaves it in the process slot. The next file
    must not quietly write into the failed file's folder.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    results_root = tmp_path / "results"
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(results_root))
    from imajin.result_bundles import reset_process_bundle

    reset_process_bundle()
    blank_source = raw / "blank.lsm"
    blank_source.write_bytes(b"stub")
    viewer.add_image(np.zeros((256, 256), dtype=np.float32), name="Ch2-T1")
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(blank_source)
    # A constant image gives segmentation nothing to threshold.
    try:
        workflows.analyze_target_cells(target="Ch2-T1", rerun=True)
    except Exception:
        pass

    good_source = raw / "real.lsm"
    good_source.write_bytes(b"stub")
    viewer.layers.clear()
    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(good_source)
    res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)

    assert res["ok"] is True
    assert res["bundle_created"] is True, "adopted the failed file's folder"
    assert "real" in Path(res["result_bundle_path"]).name
    reset_process_bundle()


def test_same_stem_in_two_folders_stays_two_samples(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """condition-per-subfolder (raw/a/x.lsm, raw/b/x.lsm) must not merge.

    run_context.samples and combined.csv are both keyed by sample_name, so
    disambiguating only the filename on disk left two different files sharing
    one record and one label in the pooled table.
    """
    import pandas as pd

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools

    reset_process_bundle()
    bundle = Path(bundle_tools.start_analysis("session")["bundle_path"])

    for condition in ("a", "b"):
        folder = tmp_path / "raw" / condition
        folder.mkdir(parents=True)
        source = folder / "rectum_1.lsm"
        source.write_bytes(b"stub")
        viewer.layers.clear()
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)
        assert workflows.analyze_target_cells(target="Ch2-T1", rerun=True)["ok"]

    meta = json.loads((bundle / "metadata.json").read_text())
    samples = meta["run_context"]["samples"]
    assert meta["run_context"]["n_samples"] == 2, "the two files merged"
    assert len({s["sample_name"] for s in samples}) == 2
    assert len({s["source_file"] for s in samples}) == 2

    combined = pd.read_csv(bundle / "tables" / "combined.csv")
    assert combined["sample_name"].nunique() == 2
    reset_process_bundle()


def test_reopening_an_adhoc_bundle_makes_it_a_real_append_target(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """open_result_bundle promises an append target even for an ad-hoc folder.

    The reuse guard rejects kind == "adhoc" as well as a closed status, so
    flipping only the status left the promise false: the analysis minted its own
    folder and evicted the bundle the resume scope still pointed at.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import ensure_active_bundle, reset_process_bundle
    from imajin.tools import bundle_resume

    reset_process_bundle()
    adhoc = ensure_active_bundle()
    assert adhoc.name.endswith("_adhoc")
    reset_process_bundle()

    bundle_resume.open_result_bundle(str(adhoc))

    source = raw / "rectum_1.lsm"
    source.write_bytes(b"stub")
    viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
    viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)
    res = workflows.analyze_target_cells(target="Ch2-T1", rerun=True)

    assert res["ok"] is True
    assert Path(res["result_bundle_path"]) == adhoc
    assert res["bundle_created"] is False
    reset_process_bundle()


def test_session_bundle_lands_next_to_the_data_in_documented_order(
    viewer,
    tmp_path,
    monkeypatch,
) -> None:
    """register_files -> start_analysis puts the folder beside the .lsm files.

    start_analysis derives its location from the registered files, so calling it
    before them lands the session bundle in the default results folder while the
    data sits elsewhere — which is how the reported session ended up with its
    bundle on C: and its images on D:.
    """
    data = tmp_path / "260818"
    data.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))
    from imajin import session as state
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools

    reset_process_bundle()
    stems = ["rectum_1", "rectum_2"]
    for stem in stems:
        source = data / f"{stem}.lsm"
        source.write_bytes(b"stub")
        state.put_file(str(source), f"{stem}.lsm")

    session_bundle = Path(bundle_tools.start_analysis("260818_roi")["bundle_path"])
    assert session_bundle.parent == data.resolve()

    for stem in stems:
        viewer.layers.clear()
        viewer.add_image(_blob_image(), name="Ch2-T1", scale=(0.5, 0.5))
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(data / f"{stem}.lsm")
        assert workflows.analyze_target_cells(target="Ch2-T1", rerun=True)["ok"]

    folders = sorted(p.name for p in data.iterdir() if p.is_dir())
    assert folders == [session_bundle.name]
    assert not (tmp_path / "fallback").exists()
    reset_process_bundle()
