"""Phase-2 spec coverage: target-channel resolution, physical units, and the
high-level analyze_target_cells workflow. Segmentation here is fed real labels
from a fixture — the Cellpose-SAM call is monkeypatched so these tests stay in
the fast suite.
"""

from __future__ import annotations

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
    assert res["n_objects"] == 2
    assert res["has_physical_units"] is True
    assert res["voxel_scale"] == [0.5, 0.5]
    df = state.get_table(res["table_name"])
    assert "mean_intensity_green_target" in df.columns
    assert "area_um2" in df.columns


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

    res = workflows.analyze_target_cells()
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
