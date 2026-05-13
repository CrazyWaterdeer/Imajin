"""Phase-2 spec coverage: target-channel resolution, physical units, and the
high-level analyze_target_cells workflow. Segmentation here is fed real labels
from a fixture — the Cellpose-SAM call is monkeypatched so these tests stay in
the fast suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from imajin.agent import state
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
    from imajin.tools import segment

    def _fake_eval(self, data, **kwargs):  # noqa: ANN001
        return mask, None, None

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return mask, None, None

    monkeypatch.setattr(segment, "_get_cellpose_model", lambda *a, **kw: _FakeModel())


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
        str(tmp_path / "results" / "bundles")
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
    assert not (tmp_path / "fallback").exists()
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

    from imajin.agent.state import get_table
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
    assert meta["schema_version"] == 2
    assert run_context["kind"] == "single"
    assert run_context["tier"] == "single_tier"
    assert run_context["status"] == "complete"
    assert len(run_context["samples"]) == 1
    assert run_context["samples"][0]["status"] == "complete"
    assert (
        run_context["samples"][0]["outputs"]["labels_cells"]
        == "labels/cells/reporter.tif"
    )


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
    assert meta["schema_version"] == 2
    assert run_context["kind"] == "single"
    assert run_context["tier"] == "two_tier"
    assert run_context["status"] == "complete"
    assert len(run_context["samples"]) == 1
    assert (
        run_context["samples"][0]["outputs"]["labels_domain"]
        == "labels/domain/reporter.tif"
    )


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
    meta = json.loads((parent / "metadata.json").read_text())
    assert meta["status"] == "in_progress"
    assert "samples" not in meta


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
